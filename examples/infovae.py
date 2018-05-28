import tensorflow as tf
import utils as tu
import click
from methods.vae import Encoder, Decoder
from methods.infovae import InfoVAE


tfd = tf.contrib.distributions


class VariationalEncoder(Encoder):
    def __init__(self, z_dim=64):
        def encoder_func(x):
            fc1 = tf.layers.dense(x, 1024, activation=tf.nn.softplus)
            fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.softplus)
            mean = tf.layers.dense(fc2, z_dim, activation=tf.identity)
            logstd = tf.layers.dense(fc2, z_dim, activation=tf.identity)
            return mean, tf.exp(logstd)

        self.encoder = tf.make_template('encoder/network', lambda x: encoder_func(x))

    @property
    def vars(self):
        return [var for var in tf.global_variables() if 'encoder' in var.name]

    def sample_and_log_prob(self, x):
        loc, scale = self.encoder(x)
        qzx = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        z_0 = qzx.sample()
        return z_0, qzx.log_prob(z_0)


class VariationalDecoder(Decoder):
    def __init__(self, z_dim=64, x_dim=784):
        self.zd = z_dim

        def decoder_func(z):
            fc1 = tf.layers.dense(z, 1024, activation=tf.nn.softplus)
            fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.softplus)
            logits = tf.layers.dense(fc2, x_dim, activation=tf.identity)
            return logits

        self.decoder = tf.make_template('decoder/network', lambda x: decoder_func(x))

    def sample_and_log_prob(self, z, x):
        pz = tfd.MultivariateNormalDiag(loc=tf.zeros_like(z), scale_diag=tf.ones_like(z))
        x_ = self.decoder(z)
        pxz = tfd.Bernoulli(logits=x_)
        return x_, tf.reduce_sum(pxz.log_prob(x), axis=1), pz.log_prob(z)

    @property
    def z_dim(self):
        return self.zd

    @property
    def prior(self):
        pz = tfd.MultivariateNormalDiag(loc=tf.zeros([self.z_dim]), scale_diag=tf.ones([self.z_dim]))
        return pz.log_prob

    @property
    def likelihood(self):
        def _fn(z, x):
            x_ = self.decoder(z)
            pxz = tfd.Bernoulli(logits=x_)
            return tf.reduce_sum(pxz.log_prob(x), axis=-1)
        return _fn

    @property
    def vars(self):
        return [var for var in tf.global_variables() if 'decoder' in var.name]


@click.command()
@click.option('--z_dim', type=click.INT, default=64)
@click.option('--mi', type=click.FLOAT, default=0.0)
@click.option('--e1', type=click.FLOAT, default=1.0)
@click.option('--e2', type=click.FLOAT, default=0.0)
@click.option('--test', is_flag=True, flag_value=True)
def main(z_dim, mi, e1, e2, test):
    test_bool = test
    from utils import find_avaiable_gpu
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_id = find_avaiable_gpu()
    print('Using device {}'.format(device_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    train, test = tu.read_mnist(batch_sizes=(250, 250))
    datasets = tu.Datasets(train=train, test=test)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.999)
    encoder = VariationalEncoder(z_dim=z_dim)
    decoder = VariationalDecoder(z_dim=z_dim)
    logdir = tu.obtain_log_path('infovae/v1/{}/{}-{}-{}/'.format(z_dim, mi, e1, e2))

    vae = InfoVAE(encoder, decoder, datasets, optimizer, logdir, mi, e1, e2)
    if not test_bool:
        vae.train(num_epochs=5000)
    vae.test()


if __name__ == '__main__':
    main()
