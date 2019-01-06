from network.YoloV2 import YoloV2
from model.dection import dectionYolov2
import numpy as np

classes_path = r'cfg/coco_classes.txt'
with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

anchor_path = r'cfg/coco_anchors.txt'
anchors = np.loadtxt(anchor_path,delimiter=',').reshape((-1,2))

net = YoloV2((608, 608, 3),box_num=len(anchors),class_num=len(class_names))
dectionYolov2(net,r'dataset/yolo_test_data',class_names,anchors,'model_saved/yolov2/model.ckpt')
