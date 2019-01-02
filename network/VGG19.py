import tensorflow as tf
import numpy as np
from model.train import fit
from network.ConvBlock import conv_block


def VGG19(input_shape):
    iw, ih, c = input_shape
    net = tf.Graph()
    with net.as_default():
        x = tf.placeholder(tf.float32, shape=(None, iw, ih, c), name='x')
        y = tf.placeholder(tf.int32, name='y')

        block1 = conv_block(x,64,(3,3),(1,1),'same',2)

        block2 = conv_block(block1, 128, (3, 3), (1, 1), 'same', 2)

        block3 = conv_block(block2, 256, (3, 3), (1, 1), 'same', 4)

        block4 = conv_block(block3, 512, (3, 3), (1, 1), 'same', 4)

        block5 = conv_block(block4, 512, (3, 3), (1, 1), 'same', 4)

        flatten = tf.layers.flatten(block5)

        fc1 = tf.layers.dense(flatten, 4096, activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        dp1 = tf.layers.dropout(fc1, 0.5)
        fc2 = tf.layers.dense(dp1, 4096, activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        dp2 = tf.layers.dropout(fc2, 0.5)

        logits = tf.layers.dense(dp2, 10,kernel_initializer=tf.contrib.layers.xavier_initializer())

        y_hat = tf.nn.softmax(logits, name='y_hat')

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y, depth=10), logits=logits)
        loss = tf.reduce_mean(loss)

        net.add_to_collection('input', {'x': x, 'y': y})
        net.add_to_collection('loss', {'loss': loss})
        net.add_to_collection('output', {'y_hat': y_hat})
        return net


if __name__ == '__main__':

    net = VGG19((224,224,3))
    X = np.random.rand(100,224,224,3)
    Y = np.random.randint(low=0,high=10,size=(100,1))
    fit(net,X,Y,10,2)