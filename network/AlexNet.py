import tensorflow as tf
import numpy as np
from model.train import fit

def AlexNet(input_shape):
    iw, ih, c = input_shape
    net = tf.Graph()
    with net.as_default():
        x = tf.placeholder(tf.float32, shape=(None, iw, ih, c), name='x')
        y = tf.placeholder(tf.int32, name='y')

        conv1 = tf.layers.conv2d(x,96,(11,11),(4,4),padding='valid',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool1 = tf.layers.max_pooling2d(conv1,(3,3),(2,2))

        conv2 = tf.layers.conv2d(pool1, 256, (5, 5), (1, 1), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv2, (3, 3), (2, 2))

        conv3 = tf.layers.conv2d(pool2, 384, (3, 3), (1, 1), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

        conv4 = tf.layers.conv2d(conv3, 384, (3, 3), (1, 1), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

        conv5 = tf.layers.conv2d(conv4, 256, (3, 3), (1, 1), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

        pool3 = tf.layers.max_pooling2d(conv5, (3, 3), (2, 2))

        flatten = tf.layers.flatten(pool3)

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

    net = AlexNet((227,227,3))
    X = np.random.rand(100,227,227,3)
    Y = np.random.randint(low=0,high=10,size=(100,1))
    fit(net,X,Y,10,2)