from network.YoloV1 import YoloV1
from model.dection import dection

voc_labels = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]

net = YoloV1((448, 448, 3))
dection(net,'dataset/yolo_test_data',voc_labels,r'model_saved/yolov1/model.ckpt')
