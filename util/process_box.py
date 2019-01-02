import tensorflow as tf
import numpy as np


def compute_area(boxes):
    w = boxes[...,3] - boxes[...,1]
    w = w * tf.cast(w > 0,dtype=tf.float32)
    h = boxes[...,2] - boxes[...,0]
    h = h * tf.cast(h > 0,dtype=tf.float32)
    overlap_area = w * h
    return overlap_area


def compute_IOU(boxes_coord_true,boxes_coord_hat,eplson=1e-6):
    size = int(np.sqrt(int(boxes_coord_true.shape[1])))
    boxes_true = yolo_boxes_to_corners(boxes_coord_true,size)
    boxes_true = tf.concat([boxes_true,boxes_true],axis=-2)

    boxes_hat = yolo_boxes_to_corners(boxes_coord_hat, size)
    overlap_top = tf.expand_dims(tf.reduce_max(tf.concat([boxes_true[...,0:1],boxes_hat[...,0:1]],axis=-1),axis=-1),axis=-1)
    overlap_left = tf.expand_dims(tf.reduce_max(tf.concat([boxes_true[..., 1:2], boxes_hat[..., 1:2]], axis=-1),axis=-1),axis=-1)
    overlap_bottom = tf.expand_dims(tf.reduce_min(tf.concat([boxes_true[..., 2:3], boxes_hat[..., 2:3]], axis=-1),axis=-1),axis=-1)
    overlap_right = tf.expand_dims(tf.reduce_min(tf.concat([boxes_true[..., 3:4], boxes_hat[..., 3:4]], axis=-1),axis=-1),axis=-1)
    boxes_overlap = tf.concat([overlap_top,overlap_left,overlap_bottom,overlap_right],axis=-1)

    boxes_true_area =  compute_area(boxes_true)
    boxes_hat_area = compute_area(boxes_hat)
    overlap_area = compute_area(boxes_overlap)
    IOU = overlap_area/(boxes_true_area+boxes_hat_area-overlap_area+eplson)
    return IOU


def corner_to_yolo_box(box,image_size):
    iw,ih = image_size
    frame_w = iw / 7
    frame_h = ih / 7
    left = box[0]
    top = box[1]
    right = box[2]
    bottom = box[3]

    x = (left+right)/2.0
    frame_x = int(x/frame_w)
    delta_x = (x-frame_x*frame_w)/frame_w
    y = (top + bottom) / 2.0
    frame_y = int(y / frame_h)
    delta_y = (y - frame_y * frame_h)/frame_h
    w = np.sqrt((right-left)/iw)
    h = np.sqrt((bottom-top)/ih)
    yolo_box = [delta_x,delta_y,w,h]
    return yolo_box,frame_x,frame_y


def yolo_boxes_to_corners(boxes_cord,size):
    """Convert YOLO box predictions to bounding box corners."""
    ind = np.reshape(np.asarray(range(size**2), dtype=np.float32), (1, size**2,1,1))
    boxes_x = (boxes_cord[..., 0:1] + ind % size) / size
    boxes_y = (boxes_cord[..., 1:2] + ind // size) / size
    boxes_w = boxes_cord[..., 2:3] ** 2
    boxes_h = boxes_cord[..., 3:4] ** 2
    left = boxes_x - boxes_w / 2.
    top = boxes_y - boxes_h / 2.
    right = boxes_x + boxes_w / 2.
    bottom = boxes_y + boxes_h / 2.
    boxes = tf.concat((
        top,  # y_min
        left,  # x_min
        bottom,  # y_max
        right  # x_max
    ),axis=-1)
    return boxes


def rescale_box(box,image_size):
    ih,iw,_ = image_size
    top = int(box[0] * ih)
    left = int(box[1] * iw)
    bottom = int(box[2] * ih)
    right = int(box[3] * iw)
    if left < 0:  left = 0
    if right > iw - 1: right = iw - 1
    if top < 0:   top = 0
    if bottom > ih - 1:   bottom = ih - 1
    return left, right, top, bottom