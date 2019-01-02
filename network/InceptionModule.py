import tensorflow as tf
from network.ConvBlock import conv2d_bn


def InceptionModule_V1(input,conv1_n,conv3_reduce,conv3_n,conv5_reduce,conv5_n,pool_proj):
    conv1 = tf.layers.conv2d(input, conv1_n, (1, 1), (1, 1), padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    conv3 = tf.layers.conv2d(input, conv3_reduce, (1, 1), (1, 1), padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv3 = tf.layers.conv2d(conv3, conv3_n, (3, 3), (1, 1), padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    conv5 = tf.layers.conv2d(input, conv5_reduce, (1, 1), (1, 1), padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv5 = tf.layers.conv2d(conv5, conv5_n, (5, 5), (1, 1), padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool = tf.layers.max_pooling2d(input, (3, 3), (1, 1),padding='same')
    pool = tf.layers.conv2d(pool, pool_proj, (1, 1), (1, 1), padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    output = tf.concat([conv1,conv3,conv5,pool],axis=-1)

    return output


def InceptionModule_V2(input,conv1_n,conv3_reduce,conv3_n,conv3_3_reduce,conv3_3_n,pool_proj):
    conv1 = conv2d_bn(input, conv1_n, (1, 1), (1, 1), padding='same')

    conv3 = conv2d_bn(input, conv3_reduce, (1, 1), (1, 1), padding='same')
    conv3 = conv2d_bn(conv3, conv3_n, (3, 3), (1, 1), padding='same')

    conv3_3 = conv2d_bn(input, conv3_3_reduce, (1, 1), (1, 1), padding='same')
    conv3_3 = tf.layers.conv2d(conv3_3, conv3_3_n, (3, 3), (1, 1), padding='same')
    conv3_3 = tf.layers.conv2d(conv3_3, conv3_3_n, (3, 3), (1, 1), padding='same')

    pool = tf.layers.max_pooling2d(input, (3, 3), (1, 1),padding='same')
    pool = conv2d_bn(pool, pool_proj, (1, 1), (1, 1), padding='same')

    output = tf.concat([conv1,conv3,conv3_3,pool],axis=-1)

    return output


def aux_classifier(input):
    pool = tf.layers.average_pooling2d(input, (5, 5), (3, 3), padding='same')
    conv = tf.layers.conv2d(pool, 128, (1, 1), (1, 1), padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    flatten = tf.layers.flatten(conv)
    fc = tf.layers.dense(flatten, 1024, activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    dp = tf.layers.dropout(fc,0.7)
    logits = tf.layers.dense(dp, 10, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return logits