from methods.vae import *
import utils as tu


class InfoVAE(VariationalAutoEncoder):
    def __init__(self, encoder, decoder, datasets, optimizer, logdir, mi, e1, e2):
        self.e1 = e1
        self.e2 = e2
        self.mi = mi
        super(InfoVAE, self).__init__(encoder, decoder, datasets, optimizer, logdir)

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
        self.l1, self.l2 = self.e1, self.e2
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

    def _create_summary(self):
        with tf.name_scope('train'):
            self.train_summary = tf.summary.merge([
                tf.summary.scalar('elbo', self.elbo),
                tf.summary.scalar('nll', self.nll),
                tf.summary.scalar('mmd', self.mmd),
                tf.summary.scalar('loss', self.loss),
                tf.summary.scalar('vae_loss', self.vae_loss)
            ])

    def _create_evaluation(self, encoder, decoder):
        x = self.iterator.get_next()
        self.z_mi = encoder.sample_and_log_prob(x)[0]
        self.log_q_z_x = encoder.sample_and_log_prob(x)[1]

    def _train(self):
        self._debug()
        self.sess.run([self.trainer])

    def _log(self, it):
        if it % 1000 == 0:
            self._logger([self.elbo, self.nll, self.vae_loss, self.loss],
                         ['elbo', 'nll', 'vae_loss', 'loss'])
            self.summary_writer.add_summary(self.sess.run(self.train_summary), it)

    def test(self):
        self._evaluate_over_test_set(
            [self.elbo, self.nll, self.mmd],
            ['elbo', 'nll', 'mmd']
        )
        self._evaluate_over_test_set(
            [self.elbo, self.nll, self.mmd],
            ['elbo', 'nll', 'mmd'], train=True
        )
