from network.YoloV1 import YoloV1
from model.train import fit
from util.data_util import load_data,preprocess_data
import tensorflow as tf

voc_labels = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]

net = YoloV1((448, 448, 3))
trainable_variables = net.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[-2:-1]
images,labels = load_data('dataset/yolo_train_data',mode='train')
X,Y=preprocess_data(images,labels,mode='train')
fit(net,X,Y,10
    ,200,trainable_list=trainable_variables,
    pretrained_weight=r'E:\Myproject\python\yolov1\model\yolov1/model.ckpt',
    save_model_dir='model_saved/yolov1/model.ckpt',lr=1e-4)