import tensorflow as tf
import numpy as np
import time
from utils import logger, gpu_session
import pickle as pkl
import os


tfd = tf.contrib.distributions


class VariationalAutoEncoder(object):
    def __init__(self, encoder, decoder, datasets, optimizer, logdir):
        self.encoder = encoder
        self.decoder = decoder
        self.datasets = datasets
        self.optimizer = optimizer
        self._create_datasets()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()
        self._create_evaluation(encoder, decoder)
        self._create_session(logdir)
        logger.configure(logdir, format_strs=['stdout', 'log'])

    def _create_datasets(self):
        datasets = self.datasets
        self.iterator = iterator = tf.data.Iterator.from_structure(
            output_types=datasets.train.output_types, output_shapes=datasets.train.output_shapes
        )
        self.train_init = iterator.make_initializer(datasets.train)
        self.test_init = iterator.make_initializer(datasets.test)

    def _create_loss(self):
        self.x = self.iterator.get_next()
        z, logqzx = self.encoder.sample_and_log_prob(self.x)
        x_, logpxz, logpz = self.decoder.sample_and_log_prob(z, self.x)

        self.encoder_loss = logqzx - logpz
        self.decoder_loss = -logpxz
        self.nll = tf.reduce_mean(self.decoder_loss)
        self.elbo = tf.reduce_mean(self.encoder_loss)
        self.loss = self.nll + self.elbo

    def _create_optimizer(self):
        encoder, decoder, optimizer = self.encoder, self.decoder, self.optimizer
        encoder_grads_and_vars = optimizer.compute_gradients(self.loss, encoder.vars)
        decoder_grads_and_vars = optimizer.compute_gradients(self.loss, decoder.vars)

        self.trainer = tf.group(optimizer.apply_gradients(encoder_grads_and_vars),
                                optimizer.apply_gradients(decoder_grads_and_vars))

    def _create_scalar_summaries(self, keys, strs):
        with tf.name_scope('train'):
            self.train_summary = tf.summary.merge([
                tf.summary.scalar(s, k) for s, k in zip(strs, keys)
            ])

    def _create_summary(self):
        self._create_scalar_summaries([self.elbo, self.nll, self.loss], ['elbo', 'nll', 'loss'])

    def _create_evaluation(self, encoder, decoder):
        pass

    def _create_session(self, logdir):
        self.summary_writer = tf.summary.FileWriter(logdir=logdir)
        self.sess = gpu_session()
        self.saver = tf.train.Saver()
        self.logdir = logdir

    def _update_optimizer(self):
        pass

    def _debug(self):
        pass

    def _train(self):
        self._debug()
        self.sess.run([self.trainer])

    def _logger(self, keys, strs):
        values = self.sess.run(keys)
        for s, v in zip(strs, values):
            logger.logkv(s, v)
        logger.dumpkvs()

    def _log(self, it):
        if it % 200 == 0:
            logger.logkv('Iteration', it)
            self._logger([self.elbo, self.nll, self.loss], ['elbo', 'nll', 'loss'])
            self.summary_writer.add_summary(self.sess.run(self.train_summary), it)

    def train(self, num_epochs):
        self.sess.run(tf.global_variables_initializer())
        it = 0
        for epoch in range(num_epochs):
            self.sess.run(self.train_init)
            self._update_optimizer()
            while True:
                try:
                    self._train()
                    it += 1
                    self._log(it)
                except tf.errors.OutOfRangeError:
                    break
        self.saver.save(sess=self.sess, save_path=self.logdir)

    def test(self):
        self.saver.restore(sess=self.sess, save_path=self.logdir)
        self._evaluate_over_test_set([self.loss, self.nll, self.elbo], ['loss', 'nll', 'elbo'])

    def _evaluate_over_test_set(self, keys, strs, train=False):
        if train:
            self.sess.run(self.train_init)
            app = 'train_'
        else:
            self.sess.run(self.test_init)
            app = 'test_'
        d = {app + s: [] for s in strs}
        while True:
            try:
                ks = self.sess.run(keys)
                for i in range(len(keys)):
                    d[app + strs[i]].append(ks[i])
            except tf.errors.OutOfRangeError:
                break
        for k in d.keys():
            d[k] = np.mean(d[k])
        self._write_evaluation(d)

    def _write_evaluation(self, d):
        logger.logkvs(d)
        logger.dumpkvs()
        try:
            with open(os.path.join(self.logdir, 'eval.pkl'), 'rb') as f:
                d_ = pkl.load(f)
        except FileNotFoundError:
            d_ = {}

        for k in d_.keys():
            if k not in d:
                d[k] = d_[k]

        with open(os.path.join(self.logdir, 'eval.pkl'), 'wb') as f:
            pkl.dump(d, f)


class Encoder(object):
    def sample_and_log_prob(self, x):
        raise NotImplementedError

    @property
    def vars(self):
        raise NotImplementedError


class Decoder(object):
    def sample_and_log_prob(self, z, x):
        raise NotImplementedError

    @property
    def prior(self):
        raise NotImplementedError

    @property
    def likelihood(self):
        raise NotImplementedError

    @property
    def z_dim(self):
        raise NotImplementedError

    @property
    def vars(self):
        raise NotImplementedError