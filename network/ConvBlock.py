import tensorflow as tf


def conv_block(input,filters,kernal_size,strides,padding,num):

    for i in range(num):
        conv = tf.layers.conv2d(input, filters, kernal_size, strides, padding=padding, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        input = conv
    pool = tf.layers.max_pooling2d(conv, (2, 2), (2, 2))
    return pool


def conv2d_bn(input,filters,kernel_size,stride,padding,activitation='relu'):
    conv = tf.layers.conv2d(input, filters, kernel_size, stride, padding=padding, activation=None,
                     kernel_initializer=tf.contrib.layers.xavier_initializer())
    bn = tf.layers.batch_normalization(conv)
    if activitation is not None:
        act = tf.nn.relu(bn)
        return act
    else:
        return bn