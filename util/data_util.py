import os
import numpy as np
import cv2
import json
from util.process_box import corner_to_yolo_box


def read_json_label(filename):
    with open(filename) as fp:
        str_json_data = fp.read()
        json_data = json.loads(str_json_data)
        label=[]
        for shape in json_data['shapes']:
            cls = np.asarray([shape['label']],dtype=np.float32)
            box = np.asarray(shape['points'],dtype=np.float32).reshape(4)
            label.append(np.concatenate([cls,box]))
        return np.asarray(label)


def load_data(path,mode='test'):
    if len(path) > 1 and os.path.isfile(path[0]):
        image_dir = ''
        file_list = path
        label_dir = os.path.join(os.path.dirname(os.path.dirname(path[0])), 'label')
    elif os.path.isdir(path):
        image_dir = os.path.join(path, 'image')
        file_list = os.listdir(image_dir)
        label_dir = os.path.join(path,'label')
    elif os.path.isfile(path):
        image_dir = ''
        file_list = [path]
        label_dir = os.path.join(os.path.dirname(os.path.dirname(path)),'label')

    image_data = []
    labels = []
    for file_name in file_list:
        if file_name.endswith('.jpg'):
            image = cv2.imread(os.path.join(image_dir, file_name))
            image_data.append(image)
            if mode=='train':
                label_filename = os.path.splitext(file_name)[0] + '.json'
                label = read_json_label(os.path.join(label_dir,label_filename))
                labels.append(label)
    if mode == 'train':
        return image_data,labels
    else:
        return image_data


def to_one_hot(label,num_class):
    one_hot_label = np.zeros(num_class)
    one_hot_label[int(label)] = 1
    return one_hot_label


def preprocess_label(label,num_class,image_size,output_size):
    labels = np.zeros((49,5+num_class))
    for obj in label:
        cls = obj[0]
        box = obj[1:]
        one_hot_label = to_one_hot(cls,num_class)
        yolo_box,frame_x,frame_y = corner_to_yolo_box(box, image_size,output_size)
        ind = frame_y*7+frame_x
        labels[ind, 0] = 1
        labels[ind, 1:5] = yolo_box
        labels[ind, 5:] = one_hot_label
    return labels


def preprocess_data(images,labels=None,target_size=(448,448),output_size=(7,7),num_class=20,mode='test'):
    image_num = len(images)
    w,h = target_size
    image_data = np.zeros((image_num,w,h,3))
    if mode == 'train':
        y_true = np.zeros((image_num,49,5+num_class))
    for i in range(image_num):
        ih,iw,_=images[i].shape
        image_data[i] = cv2.resize(images[i], target_size, interpolation=cv2.INTER_CUBIC)/255.
        if mode == 'train':
            y_true[i] = preprocess_label(labels[i], num_class, (iw,ih),output_size)
    if mode == 'train':
        return image_data,y_true
    else:
        return image_data


if __name__=='__main__':
    images,labels = load_data('../dataset/yolo_test_data',mode='train')
    print(labels)
