import tensorflow as tf
import numpy as np
from model.train import fit
from network.InceptionModule import InceptionModule_V2,aux_classifier


def InceptionV2(input_shape):
    iw, ih, c = input_shape
    net = tf.Graph()
    with net.as_default():
        x = tf.placeholder(tf.float32, shape=(None, iw, ih, c), name='x')
        y = tf.placeholder(tf.int32, name='y')

        conv1 = tf.layers.conv2d(x, 64, (7, 7), (2, 2), padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool1 = tf.layers.max_pooling2d(conv1, (3, 3), (2, 2), padding='same')

        conv2 = tf.layers.conv2d(pool1, 192, (3, 3), (1, 1), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv2, (3, 3), (2, 2), padding='same')

        inception_blok1 = InceptionModule_V2(pool2,64,96,128,16,32,32)
        inception_blok2 = InceptionModule_V2(inception_blok1, 128, 128, 192, 32, 96, 64)
        pool3 = tf.layers.max_pooling2d(inception_blok2, (3, 3), (2, 2), padding='same')

        inception_blok3 = InceptionModule_V2(pool3, 192, 96, 208, 16, 48, 64)

        inception_blok4 = InceptionModule_V2(inception_blok3, 160, 112, 224, 24, 64, 64)
        inception_blok5 = InceptionModule_V2(inception_blok4, 128, 128, 256, 24, 64, 64)
        inception_blok6 = InceptionModule_V2(inception_blok5, 112, 144, 288, 32, 64, 64)

        inception_blok7 = InceptionModule_V2(inception_blok6, 256, 160, 320, 32, 128, 128)
        pool4 = tf.layers.max_pooling2d(inception_blok7, (3, 3), (2, 2), padding='same')

        inception_blok8 = InceptionModule_V2(pool4, 256, 160, 320, 32, 128, 128)
        inception_blok9 = InceptionModule_V2(inception_blok8, 384, 192, 384, 48, 128, 128)
        pool5 = tf.layers.average_pooling2d(inception_blok9, (7, 7), (1, 1), padding='valid')
        dp = tf.layers.dropout(pool5,0.4)

        logits = tf.layers.dense(dp, 10, kernel_initializer=tf.contrib.layers.xavier_initializer())

        y_hat = tf.nn.softmax(logits, name='y_hat')

        logits1=aux_classifier(inception_blok3)
        loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y, depth=10), logits=logits1)
        loss1 = tf.reduce_mean(loss1)

        logits2 = aux_classifier(inception_blok6)
        loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y, depth=10), logits=logits2)
        loss2 = tf.reduce_mean(loss2)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y, depth=10), logits=logits)
        loss = tf.reduce_mean(loss)

        loss += 0.3*loss1+0.3*loss2

        net.add_to_collection('input', {'x': x, 'y': y})
        net.add_to_collection('loss', {'loss': loss})
        net.add_to_collection('output', {'y_hat': y_hat})

        return net


if __name__ == '__main__':
    net = InceptionV2((224,224,3))
    X = np.random.rand(100, 224, 224, 3)
    Y = np.random.randint(low=0, high=10, size=(100, 1))
    fit(net, X, Y, 10, 10,1e-4)
