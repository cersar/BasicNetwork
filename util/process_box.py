import tensorflow as tf
import numpy as np


def compute_area(boxes):
    w = boxes[...,3] - boxes[...,1]
    w = w * tf.cast(w > 0,dtype=tf.float32)
    h = boxes[...,2] - boxes[...,0]
    h = h * tf.cast(h > 0,dtype=tf.float32)
    overlap_area = w * h
    return overlap_area


def compute_IOU(boxes_true,boxes_hat,eplson=1e-6):
    boxes_true = yolo_boxes_to_corners(boxes_true)
    boxes_true = tf.concat([boxes_true,boxes_true],axis=-2)

    boxes_hat = yolo_boxes_to_corners(boxes_hat)
    overlap_top = tf.expand_dims(tf.reduce_max(tf.concat([boxes_true[...,0:1],boxes_hat[...,0:1]],axis=-1),axis=-1),axis=-1)
    overlap_left = tf.expand_dims(tf.reduce_max(tf.concat([boxes_true[..., 1:2], boxes_hat[..., 1:2]], axis=-1),axis=-1),axis=-1)
    overlap_bottom = tf.expand_dims(tf.reduce_min(tf.concat([boxes_true[..., 2:3], boxes_hat[..., 2:3]], axis=-1),axis=-1),axis=-1)
    overlap_right = tf.expand_dims(tf.reduce_min(tf.concat([boxes_true[..., 3:4], boxes_hat[..., 3:4]], axis=-1),axis=-1),axis=-1)
    boxes_overlap = tf.concat([overlap_top,overlap_left,overlap_bottom,overlap_right],axis=-1)

    boxes_true_area = compute_area(boxes_true)
    boxes_hat_area = compute_area(boxes_hat)
    overlap_area = compute_area(boxes_overlap)
    IOU = overlap_area/(boxes_true_area+boxes_hat_area-overlap_area+eplson)
    return IOU


def corner_to_yolo_box(box,image_size,output_size):
    '''lable的(left,top,right,bottom)格式转为yolo的(x,y,w,h)格式，其中(x,y,w,h)为相对于原图像的比例'''
    iw,ih = image_size
    frame_w = iw / output_size[0]
    frame_h = ih / output_size[1]
    left = box[0]
    top = box[1]
    right = box[2]
    bottom = box[3]

    x = (left+right)/2.0
    frame_x = int(x/frame_w)
    x = x/iw
    y = (top + bottom) / 2.0
    frame_y = int(y / frame_h)
    y = y/ih
    w = np.sqrt((right-left)/iw)
    h = np.sqrt((bottom-top)/ih)
    yolo_box = [x,y,w,h]
    return yolo_box,frame_x,frame_y


def yolo_boxes_to_corners(boxes):
    """Convert YOLO box predictions to bounding box corners."""
    boxes_x = boxes[..., 0:1]
    boxes_y = boxes[..., 1:2]
    boxes_w = boxes[..., 2:3]
    boxes_h = boxes[..., 3:4]
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