import tensorflow as tf
from util.print_train import *
from util.data_gen import *


def update_metrics(metrics,loss,acc,steps,batch_size,sample_num,mode='train'):
    batch_size_real = min(sample_num-(steps-1)*batch_size,batch_size)
    if mode=='train':
        metrics['loss'] += loss*batch_size_real
        if acc is not None:
            metrics['acc'] += acc*batch_size_real
    elif mode=='valid':
        metrics['val_loss'] += loss * batch_size_real
        if acc is not None:
            metrics['val_acc'] += acc * batch_size_real
    return (steps-1)*batch_size+batch_size_real


def compute_acc(pred,label):
    return np.mean(pred==label)


def fit(net,x,y,batch_size,epoch_num,lr=0.001,val_x=None,val_y=None,trainable_list=None,pretrained_weight=None,save_model_dir=None,metric_acc=False):
    train_num = x.shape[0]
    batches_per_epoch = int(np.ceil(x.shape[0]/batch_size))
    do_valid = False
    val_num = None
    if val_x is not None and val_y is not None:
        val_num = val_x.shape[0]
        do_valid = True
    with net.as_default():
        with tf.Session() as sess:
            if pretrained_weight is not None:
                saver = tf.train.Saver()
                saver.restore(sess,pretrained_weight)
            X = net.get_collection(name='input')[0]['x']
            Y = net.get_collection(name='input')[0]['y']
            loss = net.get_collection(name='loss')[0]['loss']
            y_hat = net.get_collection(name='output')[0]['y_hat']
            opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=trainable_list)

            unitialized_vars = []
            for var in tf.global_variables():
                if not sess.run(tf.is_variable_initialized(var)):
                    unitialized_vars.append(var)
            init = tf.initialize_variables(unitialized_vars)
            sess.run(init)

            for i in range(epoch_num):
                print_epoch_info(i + 1, epoch_num)
                metrics_total = {'loss': 0.0, 'val_loss': None}
                if metric_acc:
                    metrics_total['acc'] = 0.0
                    metrics_total['val_acc']=None
                for j in range(batches_per_epoch):
                    batch_x, batch_y = gen_batch_data(x, y, batch_size, j+1,shuffle=True)
                    _, train_batch_loss,pred = sess.run([opt,loss,y_hat], feed_dict={X: batch_x, Y: batch_y})
                    if metric_acc:
                        train_batch_acc = compute_acc(np.argmax(pred,axis=-1),batch_y)
                    else:
                        train_batch_acc = None
                    current = update_metrics(metrics_total, train_batch_loss, train_batch_acc, j + 1, batch_size,train_num)
                    if j + 1 == batches_per_epoch:
                        if do_valid:
                            metrics_total['val_loss']=0.0
                            metrics_total['val_acc']=0.0
                            batch_num_val = int(np.ceil(val_x.shape[0]/batch_size))
                            for k in range(batch_num_val):
                                batch_val_x, batch_val_y = gen_batch_data(val_x, val_y, batch_size, k + 1)
                                batch_val_loss,pred = sess.run([loss,y_hat], feed_dict={X: batch_val_x, Y: batch_val_y})
                                if metric_acc:
                                    batch_val_acc = compute_acc(np.argmax(pred, axis=-1), batch_val_y)
                                else:
                                    batch_val_acc = None
                                update_metrics(metrics_total, batch_val_loss, batch_val_acc, k + 1,batch_size, val_num,mode='valid')
                    print_bar(j + 1, batches_per_epoch)
                    print_metrics(metrics_total, current,val_num)

            if save_model_dir is not None:
                # tf.saved_model.simple_save(sess,save_model_dir,inputs={"X": X, "Y": Y},outputs={"y_hat": y_hat})
                saver = tf.train.Saver()
                saver.save(sess,save_model_dir)