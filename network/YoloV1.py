import tensorflow as tf
from util.process_box import compute_IOU


def YoloV1(input_shape,class_num=20,box_num=2):
    iw, ih, c = input_shape
    net = tf.Graph()
    with net.as_default():
        x = tf.placeholder(tf.float32, shape=(None, iw, ih, c), name='x')

        pad1 = tf.keras.layers.ZeroPadding2D((3,3))(x)
        leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        conv1 = tf.layers.conv2d(pad1, 64, (7, 7), (2, 2),padding='valid', activation=leaky_relu)

        pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')

        conv2 = tf.layers.conv2d(pool1, 192, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')

        conv3 = tf.layers.conv2d(pool2, 128, (1, 1), (1, 1), padding='same', activation=leaky_relu)
        conv4 = tf.layers.conv2d(conv3, 256, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        conv5 = tf.layers.conv2d(conv4, 256, (1, 1), (1, 1), padding='same', activation=leaky_relu)
        conv6 = tf.layers.conv2d(conv5, 512, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        pool3 = tf.layers.max_pooling2d(conv6, (2, 2), (2, 2), padding='same')

        conv7 = tf.layers.conv2d(pool3, 256, (1, 1), (1, 1), padding='same', activation=leaky_relu)
        conv8 = tf.layers.conv2d(conv7, 512, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        conv9 = tf.layers.conv2d(conv8, 256, (1, 1), (1, 1), padding='same', activation=leaky_relu)
        conv10 = tf.layers.conv2d(conv9, 512, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        conv11 = tf.layers.conv2d(conv10, 256, (1, 1), (1, 1), padding='same', activation=leaky_relu)
        conv12 = tf.layers.conv2d(conv11, 512, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        conv13 = tf.layers.conv2d(conv12, 256, (1, 1), (1, 1), padding='same', activation=leaky_relu)
        conv14 = tf.layers.conv2d(conv13, 512, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        conv15 = tf.layers.conv2d(conv14, 512, (1, 1), (1, 1), padding='same', activation=leaky_relu)
        conv16 = tf.layers.conv2d(conv15, 1024, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        pool4 = tf.layers.max_pooling2d(conv16, (2, 2), (2, 2), padding='same')

        conv17 = tf.layers.conv2d(pool4, 512, (1, 1), (1, 1), padding='same', activation=leaky_relu)
        conv18 = tf.layers.conv2d(conv17, 1024, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        conv19 = tf.layers.conv2d(conv18, 512, (1, 1), (1, 1), padding='same', activation=leaky_relu)
        conv20 = tf.layers.conv2d(conv19, 1024, (3, 3), (1, 1), padding='same', activation=leaky_relu)

        conv21 = tf.layers.conv2d(conv20, 1024, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        pad2 = tf.keras.layers.ZeroPadding2D((1,1))(conv21)
        conv22 = tf.layers.conv2d(pad2, 1024, (3, 3), (2, 2), padding='valid', activation=leaky_relu)
        conv23 = tf.layers.conv2d(conv22, 1024, (3, 3), (1, 1), padding='same', activation=leaky_relu)
        conv24 = tf.layers.conv2d(conv23, 1024, (3, 3), (1, 1), padding='same', activation=leaky_relu)

        flatten = tf.layers.flatten(tf.transpose(conv24,[0,3,1,2]))

        dense1 = tf.layers.dense(flatten,4096,activation=leaky_relu)
        dropout = tf.layers.dropout(dense1,rate=0.5)
        output_shape = int(conv24.shape[1])**2*(box_num*5+class_num)
        dense2 = tf.layers.dense(dropout, output_shape, activation=None)
        y = tf.placeholder(tf.float32, shape=(None, int(conv24.shape[1])**2, 1+4+class_num), name='y')

        probes, confs, boxes_cord = predict_layer(dense2,class_num,conv24.shape[1],box_num)

        y_hat = (probes, confs, boxes_cord)
        loss = compute_loss(y,y_hat)

        net.add_to_collection('input', {'x': x, 'y': y})
        net.add_to_collection('output', {'y_hat': y_hat})
        net.add_to_collection('loss', {'loss': loss})

        return net


def predict_layer(net_out,num_class,size,box_num):
    size = int(size)
    size_total = pow(size,2)
    probe_size = size_total*num_class
    conf_size = size_total*box_num
    probs = tf.reshape(net_out[:, 0:probe_size],(-1,size_total,num_class))
    confs = tf.reshape(net_out[:, probe_size:probe_size+conf_size],(-1,size_total,box_num))
    boxes_cord = tf.reshape(net_out[:, probe_size+conf_size:],(-1,size_total,box_num,4))

    return probs,confs,boxes_cord


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
