import tensorflow as tf
import numpy as np
from model.train import fit
from network.ResnetBlock import identity_block,conv_block


def ResNet50(input_shape):
    iw, ih, c = input_shape
    net = tf.Graph()
    with net.as_default():
        x = tf.placeholder(tf.float32, shape=(None, iw, ih, c), name='x')
        y = tf.placeholder(tf.int32, name='y')

        conv1 = tf.layers.conv2d(x, 64, (7, 7), (2, 2), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool1 = tf.layers.max_pooling2d(conv1, (3, 3), (2, 2), padding='same')

        conv_block1 = conv_block(pool1,(3,3),(64,64,256),strides=(1, 1))
        identity_block1 = identity_block(conv_block1, (3, 3), (64, 64, 256))
        identity_block2 = identity_block(identity_block1, (3, 3), (64, 64, 256))

        conv_block2 = conv_block(identity_block2, (3, 3), (128, 128, 512))
        identity_block3 = identity_block(conv_block2, (3, 3), (128, 128, 512))
        identity_block4 = identity_block(identity_block3, (3, 3), (128, 128, 512))
        identity_block5 = identity_block(identity_block4, (3, 3), (128, 128, 512))

        conv_block3 = conv_block(identity_block5, (3, 3), (256, 256, 1024))
        identity_block6 = identity_block(conv_block3, (3, 3), (256, 256, 1024))
        identity_block7 = identity_block(identity_block6, (3, 3), (256, 256, 1024))
        identity_block8 = identity_block(identity_block7, (3, 3), (256, 256, 1024))
        identity_block9 = identity_block(identity_block8, (3, 3), (256, 256, 1024))
        identity_block10 = identity_block(identity_block9, (3, 3), (256, 256, 1024))

        conv_block4 = conv_block(identity_block10, (3, 3), (512, 512, 2048))
        identity_block11 = identity_block(conv_block4, (3, 3), (512, 512, 2048))
        identity_block12 = identity_block(identity_block11, (3, 3), (512, 512, 2048))

        pool5 = tf.layers.average_pooling2d(identity_block12, (7, 7), (1, 1), padding='valid')
        dp = tf.layers.dropout(pool5, 0.4)

        logits = tf.layers.dense(dp, 10, kernel_initializer=tf.contrib.layers.xavier_initializer())

        y_hat = tf.nn.softmax(logits, name='y_hat')

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y, depth=10), logits=logits)
        loss = tf.reduce_mean(loss)

        net.add_to_collection('input', {'x': x, 'y': y})
        net.add_to_collection('loss', {'loss': loss})
        net.add_to_collection('output', {'y_hat': y_hat})

        return net


if __name__ == '__main__':

    net = ResNet50((224,224,3))
    X = np.random.rand(100,224,224,3)
    Y = np.random.randint(low=0,high=10,size=(100,1))
    fit(net,X,Y,10,10)