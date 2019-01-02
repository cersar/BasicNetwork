from util.process_box import rescale_box,yolo_boxes_to_corners
import numpy as np
import tensorflow as tf
import time
import cv2


def process_predict(probs, confs, boxes, threshold,image_size):
    size_total,class_num = probs.shape
    box_num = confs.shape[-1]

    cls = np.argmax(probs,axis=-1)
    probs_final = confs*(np.max(probs,axis=-1).reshape(-1,1))
    inds = tf.image.non_max_suppression(boxes.reshape((size_total*box_num,4)),probs_final.reshape((size_total*box_num)),
                                        10,iou_threshold=0.3,score_threshold=threshold).eval()
    results = []
    for ind in inds:
        i = ind//box_num
        j = ind%box_num
        left, right, top, bot = rescale_box(boxes[i, j], image_size)
        results.append([cls[i], probs_final[i,j], left, right, top, bot])

    return results


def show_dection(image_data, probes, confs, boxes_cord,labels,threshold=0.3):
    boxes = yolo_boxes_to_corners(boxes_cord, int(np.sqrt(probes.shape[1]))).eval()
    for i in range(len(image_data)):
        result = process_predict(probes[i], confs[i], boxes[i], threshold, image_data[i].shape)
        start = time.time()
        draw_dection(image_data[i], result,labels)
        end = time.time()
        print(end - start)


def draw_dection(image,results,labels):
    for result in results:
        ih,iw,_ = image.shape
        thickness = (iw+ih) // 300
        predicted_class = labels[result[0]]
        score = result[1]
        title = '{} {:.2f}'.format(predicted_class, score)
        label_size = 12
        left = result[2]
        right = result[3]
        top = result[4]
        bottom = result[5]

        if top - label_size >= 0:
            text_origin = (left, top - label_size)
        else:
            text_origin = (left, top + 1)
        cv2.rectangle(image,(left, top), (right, bottom),(0,255,0), thickness)
        cv2.putText(image, title, text_origin,0, 1e-3 * ih, (255,0,0), thickness // 3)
    cv2.imshow('',image)
    cv2.waitKey(0)