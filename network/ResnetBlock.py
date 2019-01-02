import tensorflow as tf
from network.ConvBlock import conv2d_bn


def identity_block(input, kernel_size, filters):
    filters1,filters2,filters3 = filters
    conv1 = conv2d_bn(input,filters1,(1,1),(1,1),padding='valid')
    conv2 = conv2d_bn(conv1, filters2, kernel_size, (1, 1), padding='same')
    conv3 = conv2d_bn(conv2, filters3, (1, 1), (1, 1), padding='valid',activitation=None)
    output = tf.nn.relu(conv3+input)
    return output


def conv_block(input, kernel_size, filters,strides=(2,2)):
    filters1,filters2,filters3 = filters
    conv1 = conv2d_bn(input,filters1,(1,1),strides,padding='valid')
    conv2 = conv2d_bn(conv1, filters2, kernel_size, (1, 1), padding='same')
    conv3 = conv2d_bn(conv2, filters3, (1, 1), (1, 1), padding='valid',activitation=None)

    conv4 = conv2d_bn(input, filters3, (1, 1), strides, padding='valid',activitation=None)
    output = tf.nn.relu(conv3+conv4)
    return output



