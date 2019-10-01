from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import pandas as pd 
import scipy as sp 
import gzip
import os
import tensorflow as tf 
import pickle
import subprocess
from tensorflow.python.platform import gfile

from .train_util import one_hot_encoder, get_next_batch



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
        print('Extracting', f.name)
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
    elif not isinstance(file_name,list):
        file_name = [file_name]

    if not isinstance(samples,list):
        samples = [samples]

    for s,fname in zip(samples,file_name): 
        #print(s.shape)
        with gzip.open(path+fname+'.gz', 'wb') as f:
            f.write(s)

    return 


def load_pkl(fpath):
    with open(fpath, 'rb') as file:
        from dnnlib.tflib import init_tf
        init_tf()
        return pickle.load(file, encoding='latin1')


def load_inception_net(fpath=None):
    if not fpath:
        url = 'https://drive.google.com/uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn'
        os.system('curl -L -O -C - '+url)
        os.system('mv uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn inception.pkl')

        fpath = './inception.pkl'
    return load_pkl(fpath) # inception_v3_features.pkl



def extract_inception_feature(data,inception,batch_size=16,num_gpus=1):
    activations = np.empty([data.shape[0], inception.output_shape[1]], dtype=np.float32)
    ii = 0
    iters = int(np.ceil(data.shape[0]/batch_size))
    for i in range(iters):
        start = ii
        x_batch,_,ii = get_next_batch(data,batch_size,ii,repeat=False)
        activations[start:ii] = inception.run(x_batch, num_gpus=num_gpus, assume_frozen=True)

    return activations


def insert_noise(data, noise_p, noise_dim=15,in_place=False):
    
    x_dim = data.shape[1:]
    data = data.reshape(data.shape[0],-1)

    m = int(data.shape[0]*noise_p)
    nx = np.random.choice(data.shape[0],size=m,replace=False)
    ny = np.random.choice(data.shape[1],size=noise_dim,replace=True)
    row = np.repeat(nx,noise_dim)
    col = np.concatenate([ny]*m)
    noise_data = data if in_place else data.copy() 
    noise_data[row,col] = 0.5 # for data scaled between [0,1]
    
    return noise_data.reshape(-1,*x_dim)


def gen_noise_samples_by_range(data, noise_p_range, noise_dim_range, in_place=False, p_step=0.01,dim_step=1,save_path='./'):

    for p in np.arange(noise_p_range[0], noise_p_range[1], p_step):
        for d in np.arange(noise_dim_range[0], noise_dim_range[1], dim_step):
            n_data = insert_noise(data,p,d,in_place)
            save_samples(save_path,[n_data],file_name=['noise_f'+str(int(p*100))+'_nd'+str(d)+'_samples'])
    return
