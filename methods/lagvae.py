from methods.vae import *
import utils as tu
import tensorflow as tf


class LagrangianVAE(VariationalAutoEncoder):
    def __init__(self, encoder, decoder, datasets, optimizer, logdir, mi, e1, e2):
        self.e1 = e1
        self.e2 = e2
        self.mi = mi
        super(LagrangianVAE, self).__init__(encoder, decoder, datasets, optimizer, logdir)

    def _create_loss(self):
        self.x = x = self.iterator.get_next()
        z, logqzx = self.encoder.sample_and_log_prob(x)
        self.z = z
        x_, logpxz, logpz = self.decoder.sample_and_log_prob(z, x)
        z_target = tfd.Normal(loc=tf.zeros_like(z), scale=tf.ones_like(z))
        self.mmd = mmd = tu.compute_mmd(z, z_target.sample()) * 10000
        self.nll = nll = tf.reduce_mean(-logpxz)
        self.elbo = elbo = tf.reduce_mean(logqzx - logpz)
        self.vae_loss = self.nll + self.elbo
        self.l1 = tf.get_variable('lambda1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        self.l2 = tf.get_variable('lambda2', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        if self.mi <= 0:
            loss = self.l1 * nll + (self.l1 - self.mi) * elbo + self.l2 * mmd - self.l1 * self.e1 - self.l2 * self.e2
        else:
            loss = (self.l1 + self.mi) * nll + self.l1 * elbo + self.l2 * mmd - self.l1 * self.e1 - self.l2 * self.e2
        self.loss = loss

    def _create_optimizer(self):
        encoder, decoder, optimizer = self.encoder, self.decoder, self.optimizer
        encoder_grads_and_vars = optimizer.compute_gradients(self.loss, encoder.vars)
        decoder_grads_and_vars = optimizer.compute_gradients(self.loss, decoder.vars)

        self.trainer = tf.group(optimizer.apply_gradients(encoder_grads_and_vars),
                                optimizer.apply_gradients(decoder_grads_and_vars))
        lambda_vars = [self.l1, self.l2]
        self.lambda_update = tf.train.RMSPropOptimizer(0.0001).minimize(-self.loss, var_list=lambda_vars)
        self.lambda_clip = tf.group(
            tf.assign(self.l1, tf.minimum(tf.maximum(self.l1, 0.001), 100.0)),
            tf.assign(self.l2, tf.minimum(tf.maximum(self.l2, 0.001), 100.0))
        )

    def _create_summary(self):
        with tf.name_scope('train'):
            self.train_summary = tf.summary.merge([
                tf.summary.scalar('elbo', self.elbo),
                tf.summary.scalar('nll', self.nll),
                tf.summary.scalar('mmd', self.mmd),
                tf.summary.scalar('loss', self.loss),
                tf.summary.scalar('lambda1', self.l1),
                tf.summary.scalar('lambda2', self.l2),
                tf.summary.scalar('vae_loss', self.vae_loss)
            ])

    def _create_evaluation(self, encoder, decoder):
        x = self.iterator.get_next()
        self.z_mi = encoder.sample_and_log_prob(x)[0]
        self.log_q_z_x = encoder.sample_and_log_prob(x)[1]

    def _train(self):
        self._debug()
        self.sess.run([self.trainer, self.lambda_update])
        self.sess.run(self.lambda_clip)

    def _log(self, it):
        if it % 1000 == 0:
            elbo, nll, mmd, l1, l2 = self.sess.run([self.elbo, self.nll, self.mmd, self.l1, self.l2])
            logger.log("Iteration %d: all %.4f nll %.4f elbo %.4f mmd %f l1 %.4f l2 %.4f" %
                       (it, elbo + nll, nll, elbo, mmd, l1, l2))
            self.summary_writer.add_summary(self.sess.run(self.train_summary), it)

    def test(self):
        self.saver.restore(sess=self.sess, save_path=self.logdir)
        self._evaluate_over_test_set(
            [self.elbo, self.nll, self.mmd, self.l1, self.l2],
            ['elbo', 'nll', 'mmd', 'l1', 'l2']
        )
        self._evaluate_over_test_set(
            [self.elbo, self.nll, self.mmd, self.l1, self.l2],
            ['elbo', 'nll', 'mmd', 'l1', 'l2'], train=True
        )
