import tensorflow as tf
from util.process_box import compute_IOU
from network.ConvBlock import conv2d_bn,bottleneck_block,bottleneck_block_v2
import numpy as np


def YoloV2(input_shape,class_num=80,box_num=5):
    iw, ih, c = input_shape
    net = tf.Graph()
    with net.as_default():
        x = tf.placeholder(tf.float32, shape=(None, iw, ih, c), name='x')
        anchors = tf.placeholder(tf.float32, shape=(box_num,2), name='anchors')
        anchors_reshape = tf.reshape(anchors,(1,1,box_num,2))
        conv1 = conv2d_bn(x,32,(3,3),(1,1),padding='same',activation='leaky_relu')
        pool1 = tf.layers.max_pooling2d(conv1,(2,2),(2,2))

        conv2 = conv2d_bn(pool1, 64, (3, 3), (1, 1), padding='same', activation='leaky_relu')
        pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2))

        conv5 = bottleneck_block(pool2, 128, 64)
        pool3 = tf.layers.max_pooling2d(conv5, (2, 2), (2, 2))

        conv8 = bottleneck_block(pool3, 256, 128)
        pool4 = tf.layers.max_pooling2d(conv8, (2, 2), (2, 2))

        conv13 = bottleneck_block_v2(pool4, 512, 256)
        pool5 = tf.layers.max_pooling2d(conv13, (2, 2), (2, 2))

        conv18 = bottleneck_block_v2(pool5, 1024, 512)

        conv19 = conv2d_bn(conv18, 1024, (3, 3), (1, 1), padding='same', activation='leaky_relu')
        conv20 = conv2d_bn(conv19, 1024, (3, 3), (1, 1), padding='same', activation='leaky_relu')

        conv21 = conv2d_bn(conv13, 64, (1, 1), (1, 1), activation='leaky_relu')
        conv21_reshaped = tf.space_to_depth(conv21, block_size=2)

        concat1 = tf.concat([conv21_reshaped,conv20],axis=-1)
        conv22 = conv2d_bn(concat1, 1024, (3, 3), (1, 1), padding='same', activation='leaky_relu')

        conv23 = tf.layers.conv2d(conv22, box_num*(5+class_num), (1, 1), (1, 1),activation=None)
        boxes, confs, probes = predict_layer(conv23, anchors_reshape, class_num, conv23.shape[1], box_num)

        y = tf.placeholder(tf.float32, shape=(None, int(conv23.shape[1]) ** 2, 5 + class_num), name='y')

        y_hat = (probes, confs, boxes)

        net.add_to_collection('input', {'x': x, 'anchors': anchors})
        net.add_to_collection('output', {'y_hat': y_hat})

        return net


def pred_cord_to_box(boxes_pred,size,anchors):
    """Convert YOLOv2 box predictions to bounding box cords(x,y,w,h)."""
    ind = np.reshape(np.asarray(range(size ** 2), dtype=np.float32), (1, size ** 2, 1, 1))
    boxes_x = (tf.sigmoid(boxes_pred[..., 0:1]) + ind % size) / size
    boxes_y = (tf.sigmoid(boxes_pred[..., 1:2]) + ind // size) / size
    boxes_wh = tf.exp(boxes_pred[..., 2:4])*anchors / size
    boxes_w = boxes_wh[..., :1]
    boxes_h = boxes_wh[..., 1:2]

    boxes = tf.concat((
        boxes_x,
        boxes_y,
        boxes_w,
        boxes_h
    ), axis=-1)
    return boxes


def predict_layer(net_out,anchors,num_class,size,box_num):
    size=int(size)
    net_out = tf.reshape(net_out,(-1,size**2,box_num,5+num_class))
    boxes_pred = net_out[..., :4]
    boxes = pred_cord_to_box(boxes_pred,size,anchors)
    box_confidence = tf.sigmoid(net_out[..., 4:5])
    box_class_probs = tf.nn.softmax(net_out[..., 5:])
    return boxes,box_confidence,box_class_probs


def compute_loss(y_true,y_hat,lambd_coord=5,lambd_nonObj=.5):
    probes_hat, confs_hat, boxes_cord_hat = y_hat
    obj_mask = y_true[..., 0]

    confs_true = tf.expand_dims(obj_mask,axis=2)
    boxes_cord_true = tf.expand_dims(y_true[...,1:5],axis=2)
    probes_true = y_true[...,5:]
    IOU = compute_IOU(boxes_cord_true,boxes_cord_hat)
    IOU_max = tf.reshape(tf.reduce_max(confs_hat,axis=-1),(-1,confs_hat.shape[1],1))
    box_mask = tf.cast(IOU >= IOU_max, dtype=tf.float32) * tf.reshape(obj_mask, (-1, confs_hat.shape[1], 1))

    location_loss = tf.reduce_sum(tf.reduce_sum(box_mask*tf.reduce_sum(tf.pow(boxes_cord_hat-boxes_cord_true,2),axis=-1),axis=-1),axis=-1)
    conf_diff_sum_obj = tf.reduce_sum(box_mask*tf.pow(confs_hat - confs_true,2),axis=-1)
    conf_loss_obj = tf.reduce_sum(tf.reduce_sum(conf_diff_sum_obj,axis=-1),axis=-1)
    conf_diff_sum_nonObj = tf.reduce_sum((1-box_mask) * tf.pow(confs_hat - confs_true, 2), axis=-1)
    conf_loss_nonObj = tf.reduce_sum(tf.reduce_sum(conf_diff_sum_nonObj,axis=-1),axis=-1)
    class_loss = tf.reduce_sum(obj_mask*tf.reduce_sum(tf.pow(probes_hat-probes_true,2),axis=-1),axis=-1)
    loss = tf.reduce_mean(lambd_coord*location_loss+conf_loss_obj+lambd_nonObj*conf_loss_nonObj+class_loss)
    return loss


if __name__ == '__main__':
    YoloV2((224,224,3))
