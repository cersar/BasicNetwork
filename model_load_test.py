from keras.datasets import mnist
import numpy as np
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data('dataset/mnist.npz')
x_test = x_test[:, :, :, np.newaxis] / 255.

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], 'model_saved/LeNet')
    X = sess.graph.get_tensor_by_name("x:0")
    y_hat = sess.graph.get_tensor_by_name("y_hat:0")
    pred = sess.run(y_hat, feed_dict={X: x_test[0:1, :, :, :]})
    print(np.argmax(pred))
    print(y_test[0])









