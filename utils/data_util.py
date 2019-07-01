from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import pandas as pd 
import scipy as sp 
import gzip
import os
import tensorflow as tf 
from tensorflow.python.platform import gfile

from utils.train_util import one_hot_encoder



# from tensorflow, extract MNIST style images
def extract_data(fpath,shape,dtype=np.float32):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
        f: A file object that can be passed into a gzip reader.
    Returns:
        data: A numpy array [index, y, x, depth].
    Raises:
        ValueError: If the bytestream does not start with 2051.
    """
    with open(fpath, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            buf = bytestream.read()
            data = np.frombuffer(buf, dtype=dtype)
            data = data.reshape([-1]+shape)
        return data


# from tensorflow, extract MNIST style labels
def extract_labels(fpath,shape,one_hot=False, num_classes=10,dtype=np.uint8):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
        f: A file object that can be passed into a gzip reader.
        one_hot: Does one hot encoding for the result.
        num_classes: Number of classes for the one hot encoding.
    Returns:
        labels: a 1D uint8 numpy array.
    Raises:
        ValueError: If the bystream doesn't start with 2049.
    """
    with gfile.Open(fpath, 'rb') as f:
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            #size = np.prod(shape)*np.dtype(dtype).itemsize
            buf = bytestream.read()
            labels = np.frombuffer(buf, dtype=dtype)
            if one_hot:
                return one_hot_encoder(labels, num_classes)
            labels = labels.reshape([-1]+shape)
            return labels


def save_samples(path,samples,file_name=None):
    
    if not os.path.exists(path):
        os.makedirs(path)
    if path[-1] != '/':
        path+='/'

    if file_name is None:
        file_name = ['samples','labels']

    if not isinstance(samples,list):
        samples = [samples]

    for s,fname in zip(samples,file_name): 
        #print(s.shape)
        with gzip.open(path+fname+'.gz', 'wb') as f:
            f.write(s)
