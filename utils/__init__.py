import tensorflow as tf
import GPUtil
import numpy as np
import os
from collections import namedtuple


Datasets = namedtuple('datasets', ['train', 'test'])
PREFIX = os.environ['DATA_PREFIX']


def sample_binary(x):
    r = tf.random_uniform(tf.shape(x), minval=0.0, maxval=1.0)
    return tf.cast(x > r, tf.float32)


def sample_binary_label(x, y):
    r = tf.random_uniform(tf.shape(x), minval=0.0, maxval=1.0)
    return tf.cast(x > r, tf.float32), y


def read_mnist(path='mnist', label=False, binarized=True, batch_sizes=(250, 250)):
    """
    Returns train and test sets for MNIST.
    :param path: path to read / download MNIST dataset.
    :param label: whether there is label or not.
    :param binarized: if the dataset is stochastically binarized.
    :param batch_sizes: batch sizes of train and test.
    :return: Two tf.data.Datasets, train and test.
    """
    # from .mnist import make_train, make_test
    path = os.path.join(PREFIX, path)
    # train, test = make_train(path, label), make_test(path, label)
    fn = sample_binary_label if label else sample_binary
    from tensorflow.examples.tutorials.mnist import input_data
    mnists = input_data.read_data_sets(path)
    if label:
        train = tf.data.Dataset.from_tensor_slices((mnists.train.images, mnists.train.labels))
        test = tf.data.Dataset.from_tensor_slices((mnists.test.images, mnists.test.labels))
    else:
        train = tf.data.Dataset.from_tensor_slices(mnists.train.images)
        test = tf.data.Dataset.from_tensor_slices(mnists.test.images)
    train, test = train.batch(batch_sizes[0]), test.batch(batch_sizes[1])
    if binarized:
        train, test = train.map(fn), test.map(fn)
    return train, test


def find_avaiable_gpu(max_load=0.3, max_memory=0.5):
    gpu_avail = GPUtil.getFirstAvailable(attempts=10000, maxLoad=max_load, maxMemory=max_memory, interval=199)
    return gpu_avail[0]


def gpu_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))


def obtain_log_path(path):
    import os
    prefix = os.environ['EXP_LOG_PATH']
    assert len(prefix) > 0, 'Please set environment variable EXP_LOG_PATH'
    return os.path.join(prefix, path)


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):
    """
    Compute kernel-estimated MMD between two variables, both with size [batch_size, dim]
    :param x:
    :param y:
    :return:
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)