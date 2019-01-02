from util.process_box import rescale_box,yolo_boxes_to_corners
import numpy as np
import tensorflow as tf
import time
import cv2


def process_predict(probs, confs, boxes, image_size,score_threshold=0.5,iou_threshold=0.2,max_boxes_num=10):
    size_total,class_num = probs.shape

    cls = np.argmax(probs,axis=-1)
    probs_final = confs * (np.max(probs, axis=-1).reshape(-1, 1))

    D = {i: [] for i in list(range(class_num))}
    for ind,c in enumerate(cls):
        D[c].append(ind)

    results = []
    for c,inds in D.items():
        if len(inds) == 0:
            continue
        boxes_cls = boxes[inds,...].reshape((-1, 4))
        probs_cls = np.squeeze(probs_final[inds, ...].reshape((-1, 1)))
        nms_inds = tf.image.non_max_suppression(boxes_cls, probs_cls, max_boxes_num,
                                                iou_threshold=iou_threshold, score_threshold=score_threshold).eval()
        for ind in nms_inds:
            left, right, top, bottom = rescale_box(boxes_cls[ind], image_size)
            results.append([c, probs_cls[ind], left, right, top, bottom])

    return results


def show_dection(image_data, probes, confs, boxes_cord,labels):
    boxes = yolo_boxes_to_corners(boxes_cord, int(np.sqrt(probes.shape[1]))).eval()
    for i in range(len(image_data)):
        result = process_predict(probes[i], confs[i], boxes[i],image_data[i].shape)
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