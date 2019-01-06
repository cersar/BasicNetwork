import tensorflow as tf


def conv_block(input,filters,kernal_size,strides,padding,num):

    for i in range(num):
        conv = tf.layers.conv2d(input, filters, kernal_size, strides, padding=padding, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        input = conv
    pool = tf.layers.max_pooling2d(conv, (2, 2), (2, 2))
    return pool


def conv2d_bn(input,filters,kernel_size,stride,padding='valid',activation='relu',alpha=0.1):
    conv = tf.layers.conv2d(input, filters, kernel_size, stride, padding=padding, activation=None,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(),use_bias=False)
    bn = tf.layers.batch_normalization(conv)
    if activation == 'relu':
        act = tf.nn.relu(bn)
        return act
    elif activation == 'leaky_relu':
        leaky_relu = tf.keras.layers.LeakyReLU(alpha=alpha)
        act = leaky_relu(bn)
        return act
    else:
        return bn


def bottleneck_block(input,outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
    conv1 = conv2d_bn(input,outer_filters,(3,3),(1,1),padding='same',activation='leaky_relu')
    conv2 = conv2d_bn(conv1,bottleneck_filters,(1,1),(1,1),activation='leaky_relu')
    conv3 = conv2d_bn(conv2, outer_filters, (3, 3), (1, 1), padding='same',activation='leaky_relu')
    return conv3


def bottleneck_block_v2(input,outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3convolutions."""
    conv3 = bottleneck_block(input,outer_filters, bottleneck_filters)
    conv4 = conv2d_bn(conv3,bottleneck_filters,(1,1),(1,1),activation='leaky_relu')
    conv5 = conv2d_bn(conv4, outer_filters, (3, 3), (1, 1), padding='same',activation='leaky_relu')
    return conv5