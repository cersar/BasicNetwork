from util.dection_util import show_dection
import tensorflow as tf
from util.data_util import load_data,preprocess_data
import time


def dection(net,image_path,labels,weights_file):
    with net.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, weights_file)
            start = time.time()
            image_data = load_data(image_path)
            X = preprocess_data(image_data)
            end = time.time()
            print("image load time {:.2f}".format(end - start))
            start1 = time.time()
            x = net.get_collection('input')[0]['x']
            y_hat = net.get_collection('output')[0]['y_hat']
            probes, confs, boxes_cord = sess.run(y_hat, feed_dict={x: X})
            end1 = time.time()
            print("forward time {:.2f}".format(end1 - start1))
            start2 = time.time()
            show_dection(image_data, probes, confs, boxes_cord,labels)
            end2 = time.time()
            print("show dection time {:.2f}".format(end2 - start2))
            end = time.time()
            print("dect {} iamges, total time {:.2f}".format(len(image_data),end-start))