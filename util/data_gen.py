import numpy as np


def shuffle_data(X,Y):
    ind = np.random.permutation(range(X.shape[0]))
    X = X[ind]
    Y = Y[ind]
    return X, Y


def gen_batch_data(X,Y,batch_size,current_batch,shuffle=False):
    if current_batch==1 and shuffle is True:
        X,Y=shuffle_data(X,Y)
    start = (current_batch-1)*batch_size
    end = min(current_batch*batch_size,X.shape[0])
    return X[start:end],Y[start:end]