import tensorflow as tf
import numpy as np
from model.train import fit
from keras.datasets import mnist


def LeNet(input_shape):
    iw,ih,c = input_shape
    net = tf.Graph()
    with net.as_default():
        x = tf.placeholder(tf.float32,shape=(None,iw,ih,c),name='x')
        y = tf.placeholder(tf.int32,name='y')
        conv1_W=tf.get_variable("conv1_W",shape=[5,5,1,6],initializer=tf.contrib.layers.xavier_initializer())
        conv1=tf.nn.conv2d(x,conv1_W,[1,1,1,1],padding='SAME')
        conv1_act = tf.nn.tanh(conv1)
        pool1 = tf.nn.avg_pool(conv1_act,[1,2,2,1],[1,2,2,1],padding='VALID')

        conv2_W = tf.get_variable("conv2_W", shape=[5, 5, 6, 16], initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.conv2d(pool1, conv2_W, [1, 1, 1, 1], padding='VALID')
        conv2_act = tf.nn.tanh(conv2)
        pool2 = tf.nn.avg_pool(conv2_act, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

        flatten = tf.reshape(pool2,(-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]))

        dense1_W = tf.get_variable("dense1_W",shape=[flatten.shape[1],120],initializer=tf.contrib.layers.xavier_initializer())
        dense1_b = tf.get_variable("dense1_b", shape=[1,120],initializer=tf.initializers.zeros())
        dense1 = tf.matmul(flatten,dense1_W)+dense1_b
        dense1_act = tf.nn.tanh(dense1)

        dense2_W = tf.get_variable("dense2_W", shape=[120,84],
                                   initializer=tf.contrib.layers.xavier_initializer())
        dense2_b = tf.get_variable("dense2_b", shape=[1, 84], initializer=tf.initializers.zeros())
        dense2 = tf.matmul(dense1_act,dense2_W ) + dense2_b
        dense2_act = tf.nn.tanh(dense2)

        dense3_W = tf.get_variable("dense3_W", shape=[84, 10],
                                   initializer=tf.contrib.layers.xavier_initializer())
        dense3_b = tf.get_variable("dense3_b", shape=[1, 10], initializer=tf.initializers.zeros())
        logit = tf.matmul(dense2_act,dense3_W) + dense3_b
        y_hat = tf.nn.softmax(logit,name='y_hat')

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y,depth=10),logits=logit)
        loss = tf.reduce_mean(loss)

        net.add_to_collection('input', {'x':x,'y':y})
        net.add_to_collection('loss', {'loss':loss})
        net.add_to_collection('output', {'y_hat':y_hat})
        return net


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data('../dataset/mnist.npz')
    x_train = x_train[:, :, :, np.newaxis] / 255.
    x_test = x_test[:, :, :, np.newaxis] / 255.
    net = LeNet(input_shape=(28, 28, 1))
    fit(net, x_train, y_train, 64, 10,x_test,y_test,save_model_dir='../model_saved/LeNet')

