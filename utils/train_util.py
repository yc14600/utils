
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#import edward as ed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import seaborn as sb
import numpy as np
import scipy as sp
import os
path = os.getcwd()
import sys
sys.path.append(path+'/../')

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.datasets import cifar10,cifar100
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export

def concat_parms(parm_list):
    pa = tf.reshape(parm_list[0],[1,-1])
    for i in range(1,len(parm_list)):
        p = tf.reshape(parm_list[i],[1,-1])
        pa = tf.concat([pa,p],1)
    return pa

def standardize_flatten(X_TRAIN,X_TEST,standardize=True,flatten=True):
    TRAIN_SIZE = X_TRAIN.shape[0]
    X = np.vstack((X_TRAIN,X_TEST)).astype(np.float32)
    if standardize:
        X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
    if flatten:
        X = X.reshape(X.shape[0],-1)
    X_TRAIN = X[:TRAIN_SIZE,]
    X_TEST = X[TRAIN_SIZE:,]
    return X_TRAIN,X_TEST

def gen_permuted_data(seed,x_train,x_test):
    np.random.seed(seed)
    perm_inds = np.arange(x_train.shape[1])
    np.random.shuffle(perm_inds)
    return x_train[:,perm_inds],x_test[:,perm_inds]

def gen_class_split_data_multiclass(seed,train_size,test_size,x_train,y_train,x_test,y_test,clss=None):
    np.random.seed(seed)
    if clss is None:
        clss = np.random.choice(y_test.shape[1],size=2,replace=False)
    print('select classes',clss)
    mask = np.zeros(y_test.shape[1])
    mask[clss] = 1
    train_idx = np.sum(np.abs(mask - y_train),axis=1)==1
    t_x_train = x_train[train_idx][:train_size]
    t_y_train = y_train[train_idx][:train_size]
    test_idx = np.sum(np.abs(mask - y_test),axis=1)==1
    t_x_test = x_test[test_idx][:test_size]
    t_y_test = y_test[test_idx][:test_size]

    return t_x_train,t_y_train,t_x_test,t_y_test


def split_data(x_train,y_train,clss,C,train_size,one_hot):
    train_idx = (y_train==clss[0]).reshape(-1)
    for c in range(1,C):
        #print(c,np.sum(train_idx))
        train_idx = train_idx | (y_train==clss[c]).reshape(-1)
    if  train_size is None or train_size <= x_train[train_idx].shape[0]:
        t_x_train = x_train[train_idx][:train_size]
        ty_train_tmp = y_train[train_idx][:train_size]
    else:
        # repeat samples
        rep = int(np.floor(train_size/x_train[train_idx].shape[0]))
        rd = train_size - rep*x_train[train_idx].shape[0]
        rd_ids = np.random.choice(np.arange(np.sum(train_idx)),rd,replace=False)
        #print('rd',rd,rd_ids.shape,max(rd_ids))

        t_x_train = np.repeat(x_train[train_idx],rep,axis=0)
        t_x_train = np.vstack([t_x_train,x_train[train_idx][rd_ids]])

        ty_train_tmp = np.repeat(y_train[train_idx],rep,axis=0)
        ty_train_tmp = np.concatenate([ty_train_tmp,y_train[train_idx][rd_ids]])

    if one_hot:
        t_y_train = np.zeros((ty_train_tmp.shape[0],C))       
        for c in range(C):           
            t_y_train[(ty_train_tmp==clss[c]).reshape(-1),c] = 1
    else:
        t_y_train = ty_train_tmp
           
    return t_x_train,t_y_train


def gen_class_split_data(seed,train_size,test_size,x_train,y_train,x_test,y_test,clss=None,one_hot=True,C=2):
    
    if clss is None:
        np.random.seed(seed)
        K = int(np.max(y_train)+1)
        clss = np.random.choice(K,size=C,replace=False)
    print('select classes',clss)
    #print('one hot',one_hot)
    t_x_train,t_y_train = split_data(x_train,y_train,clss,C,train_size,one_hot)
    if test_size is None or test_size > 0:
        t_x_test,t_y_test = split_data(x_test,y_test,clss,C,test_size,one_hot)
    else:
        t_x_test,t_y_test = None,None

    return t_x_train,t_y_train,t_x_test,t_y_test


def gen_class_split_data_with_noise(seed,train_size,test_size,x_train,y_train,x_test,y_test,clss=None,cls_ns=None,ns=0.1):
    np.random.seed(seed)
    if clss is None:
        clss = np.random.choice(y_test.shape[1],size=2,replace=False)
    print('select classes',clss)

    C = y_train.max()+1
    if clss_ns is None:
        cls_ns = list(range(C))

    for c in clss:
        cls_ns.remove(c)
    print('noise classes',cls_ns)

    t_x_train,t_y_train,t_x_test,t_y_test = gen_class_split_data(seed,train_size,test_size,x_train,y_train,x_test,y_test,clss,one_hot=False)
    if len(cls_ns)>0:
        noise_train_size = int(t_x_train.shape[0] * ns)
        noise_x = np.random.normal(size=(noise_train_size,x_train.shape[1])).astype(np.float32)
        noise_y = np.random.choice(cls_ns,size=noise_train_size)
        t_x_train = np.concatenate((t_x_train,noise_x),axis=0)
        t_y_train = np.concatenate((t_y_train,noise_y),axis=0)
    
    t_y_train = one_hot_encoder(t_y_train,C)
    t_y_test = one_hot_encoder(t_y_test,C)
    perm_inds = np.arange(t_x_train.shape[0])
    np.random.shuffle(perm_inds)
    t_x_train,t_y_train = t_x_train[perm_inds],t_y_train[perm_inds]
    
    return t_x_train,t_y_train,t_x_test,t_y_test,cls_ns


def get_next_batch(data, B, ii,labels=None,repeat=True):
    if B == data.shape[0] or (not repeat and B > data.shape[0]):
        return data,labels,ii
    elif B > data.shape[0]:        
        n = B/data.shape[0]
        r = B%data.shape[0]
        x_batch = np.concatenate([np.repeat(data,n,axis=0),data[:r]])

        if labels is not None:
            y_batch = np.concatenate([np.repeat(labels,n,axis=0),labels[:r]])
            return x_batch,y_batch,0
        else:
            return x_batch,None,0

    if ii+B < data.shape[0]:
        if not labels is None:
            return data[ii:ii+B],labels[ii:ii+B],ii+B
            
        else:
            return data[ii:ii+B],None,ii+B
    else:
        if repeat:
            
            r = ii+B-data.shape[0]
            ids = np.arange(data.shape[0])
            batch = data[(ids>=ii)|(ids<r)]
            if labels is None:
                return batch,None,r
            else:
                return batch,labels[(ids>=ii)|(ids<r)],r
        else:
            if labels is None:
                return data[ii:],None,0
            else:
                return data[ii:],labels[ii:],0


def plot(samples,shape=None,cmap='Greys_r'):
    if shape is None:
        rows = 4
        cols = 4
    else:
        rows = shape[0]
        cols = shape[1]
        
    fig = plt.figure(figsize=(rows, cols))
    gs = gridspec.GridSpec(rows, cols)    
    #gs.update(wspace=0.01, hspace=0.01)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.set_aspect('equal')
        #if MNIST:
        #    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        #else:
        if cmap == 'Greys_r':
            assert(len(sample.shape)<3 or sample.shape[2]==1)
            sample = sample.reshape(sample.shape[0],sample.shape[1])
        #print('plot sample shape',sample.shape)
        plt.imshow(sample,cmap=cmap)

    return fig


def load_data(dataset,mtype,N=0,M=0,K=0,s_mean=0.,s_std=1.,d_mean=0.,d_std=1.,noise_std=.1,data_dir=None,probs=None):
    D_true = None
    if dataset == 'synthesis':
        if mtype == 'dictionary':
            D_true,X = build_toy_dataset(mtype,N,M,K,s_std,s_mean,d_mean=d_mean,noise_std=noise_std)
            #normalise
            #X = util.standardize(X,each_feature=True)
            
        elif mtype == 'gaussian':
            _,X = build_toy_dataset(mtype,N,M,0,s_std,s_mean,noise_std=noise_std)   

        elif mtype == 'mix_gaussian' or mtype == 'bpm':
            D_true, X = build_toy_dataset('mix_gaussian',N,M,K,s_std=s_std,s_mean=s_mean,d_std=d_std,noise_std=noise_std,probs=probs)
        
    elif dataset in ['MNIST', 'FASHION']:
        DATA_DIR = data_dir#"../data/"+str.lower(dataset)
        #IMG_DIR = "img"
        data = input_data.read_data_sets(DATA_DIR)    
        X = data.train.images
        D_true = data.train.labels
        #ids = mnist.train.labels==4
        #X = X[ids]
        #N = X.shape[0]
        #M = X.shape[1]
        #normalise
        #X = (X - X.mean())/X.std()
    '''
    elif dataset == 'CIFAR10':
        data = unpickle(data_dir+'/cifar-10-batches-py/data_batch_1')
        X = data['data']
        D_true = data['labels']
    '''
        
    return X, D_true


def one_hot_encoder(label,H=None,out=None):

    N = label.shape[0]
    if H is None:
        H = int(np.max(label)+1)
    label = label.astype(np.int32)
    
    if out is None:
        Y = np.zeros((N,H),dtype=np.float32)
    else:
        Y = out * 0
        
    Y[range(N),label] = 1

    
    return Y

def expand_nsamples(data,n_samples):
    data_ns = np.expand_dims(data,axis=0)
    data_ns = np.repeat(data_ns,n_samples,axis=0)
    return data_ns


def build_toy_dataset(mtype,N,M,K,s_std=1,s_mean=0,d_std=1,d_mean=0,noise_std=0.1,probs=None):
    if mtype == 'dictionary':
        D = []
        #print(s_mean,s_std)
        s = np.random.normal(s_mean,s_std,(N,K))               
        for k in range(K):
            D.append(np.sin(((k+1)*2*(np.arange(0,2*np.pi,2*np.pi/M)))))
        D = np.vstack(D)
        x = np.matmul(s,D) + np.random.normal(0, noise_std, size=(N,M))
        return (D,x)
    elif mtype == 'gaussian':
        s = np.random.normal(s_mean,s_std,(N,M))
        x = s + np.random.normal(0,noise_std,size=(N,M))
        return (None,x)
    elif mtype == 'mix_gaussian':
        y = np.random.choice(K,N,p=probs)
        x = np.zeros((N,M))
        s_mean = np.zeros(M) + s_mean
        
        for k in range(K):
            mean = s_mean[k]
            std = np.ones(M) * s_std[k]
            ids = (y==k)
            #print(mean,std)
            x[ids] = np.random.normal(mean,std,(sum(ids),M))       
        return (y,x)


    
def config_optimizer(starter_learning_rate,step_name,grad_type='adam',decay=None,beta1=0.9,scope=None):

    if not scope:
        scope = step_name.split('_')[0]
    print('config optimizer, grad type {}, scope {}'.format(grad_type,scope))
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        step = tf.get_variable(initializer=0, trainable=False, name=step_name)
        if decay is not None:
            learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                step,
                                                decay[0], decay[1], staircase=True)
        else:
            learning_rate = starter_learning_rate

        if 'adam' in grad_type:                                       
            optimizer = (tf.train.AdamOptimizer(learning_rate,beta1=beta1),step)
        elif 'sgd' in grad_type:
            optimizer = (tf.train.GradientDescentOptimizer(learning_rate),step)
        elif 'ada_delta' in grad_type:
            optimizer = (tf.train.AdadeltaOptimizer(learning_rate),step)
    
    return optimizer

def gen_class_select_data(x_train,y_train,x_test,y_test,cl,one_hot_code=None):
    train_ids = (y_train==cl).reshape(-1)
    test_ids = (y_test==cl).reshape(-1)
    
    if one_hot_code is None:
        y_train_cl = y_train[train_ids]
        y_test_cl = y_test[test_ids]
    else:
        y_train_cl = np.repeat(one_hot_code.reshape(1,-1),np.sum(train_ids),axis=0)
        y_test_cl =  np.repeat(one_hot_code.reshape(1,-1),np.sum(test_ids),axis=0)
    #print('label shape',y_train_cl.shape,y_test_cl.shape)

    return x_train[train_ids],y_train_cl,x_test[test_ids],y_test_cl
    

def gen_next_task_data(task_name,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,sd=0,cl_n=2,out_dim=2,num_heads=1,cl_cmb=None,cl_k=0):

    if 'permuted' in task_name:
        x_train_task,x_test_task = gen_permuted_data(sd,X_TRAIN,X_TEST)
        y_train_task = Y_TRAIN
        y_test_task = Y_TEST
        clss = None

    elif 'cross_split' in task_name:
        x_trains,y_trains,x_tests,y_tests = [],[],[],[]
        for k in range(out_dim):
            one_hot_code = np.zeros(out_dim)
            one_hot_code[k] = 1
            x_train_k,y_train_k,x_test_k,y_test_k = gen_class_select_data(X_TRAIN[k],Y_TRAIN[k],X_TEST[k],Y_TEST[k],cl_k,one_hot_code=one_hot_code)
            x_trains.append(x_train_k)
            y_trains.append(y_train_k)
            x_tests.append(x_test_k)
            y_tests.append(y_test_k)

        x_train_task = np.vstack(x_trains)
        y_train_task = np.vstack(y_trains)
        sids = np.arange(x_train_task.shape[0])
        np.random.shuffle(sids)
        x_train_task,y_train_task = x_train_task[sids],y_train_task[sids]

        x_test_task = np.vstack(x_tests)
        y_test_task = np.vstack(y_tests)

        cl_k += 1
        clss = None

    elif 'split' in task_name:
        clss = cl_cmb[cl_k:cl_k+cl_n]
        if num_heads > 1:
            x_train_task,y_train_task,x_test_task,y_test_task = gen_class_split_data(sd,None,None,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,clss,C=cl_n)
        else:   
            x_train_task,y_train_task,x_test_task,y_test_task = gen_class_split_data(sd,None,None,X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,clss,one_hot=False,C=cl_n)
            y_train_task = one_hot_encoder(y_train_task,out_dim)
            y_test_task = one_hot_encoder(y_test_task,out_dim) 
        
        cl_k+=cl_n


    return x_train_task,y_train_task,x_test_task,y_test_task,cl_k,clss

def load_task_data(task_name,DATA_DIR,TRAIN_SIZE=5000,TEST_SIZE=1000,dataset='mnist',out_dim=10):
    if 'permuted' in task_name:
        data = input_data.read_data_sets(DATA_DIR,one_hot=True) 
        shuffle_ids = np.arange(data.train.images.shape[0])
        X_TRAIN = data.train.images[shuffle_ids][:TRAIN_SIZE]
        Y_TRAIN = data.train.labels[shuffle_ids][:TRAIN_SIZE]
        X_TEST = data.test.images[:TEST_SIZE]
        Y_TEST = data.test.labels[:TEST_SIZE]
    
    elif 'split' in task_name:
        if dataset == 'cifar10':       
            (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = cifar10.load_data() 
            # standardize data
            X_TRAIN,X_TEST = standardize_flatten(X_TRAIN,X_TEST,flatten=False)
            print('data shape',X_TRAIN.shape)

            Y_TRAIN = one_hot_encoder(Y_TRAIN.reshape(-1),out_dim)
            Y_TEST = one_hot_encoder(Y_TEST.reshape(-1),out_dim)

        elif 'mnist' in dataset:
            data = input_data.read_data_sets(DATA_DIR) 
            X_TRAIN = np.concatenate([data.train.images,data.validation.images],axis=0)
            Y_TRAIN = np.concatenate([data.train.labels,data.validation.labels],axis=0)
            X_TEST = data.test.images
            Y_TEST = data.test.labels
         
    return X_TRAIN,Y_TRAIN,X_TEST,Y_TEST

def load_mini_quick_draw(filepath):
    categories = ['fruits','vehicles','buildings','animals','instruments']
    X_TRAIN,Y_TRAIN, X_TEST, Y_TEST = [],[],[],[]
    for cn in categories:
        data = np.load(filepath+cn+'/processed.npz')
        X_TRAIN.append(data['x_train'])
        Y_TRAIN.append(data['y_train'])
        X_TEST.append(data['x_test'])
        Y_TEST.append(data['y_test'])
    return X_TRAIN,Y_TRAIN,X_TEST,Y_TEST

def set_ac_fn(ac_name):
    
    if ac_name == 'relu':
        ac_fn = tf.nn.relu
    elif ac_name == 'tanh':
        ac_fn = tf.nn.tanh
    elif ac_name == 'sigmoid':
        ac_fn = tf.nn.sigmoid
    elif ac_name == 'softmax':
        ac_fn = tf.nn.softmax
    elif ac_name == 'softplus':
        ac_fn = tf.nn.softplus

    return ac_fn

def get_var_list(scope):
    tmp = set()
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
        tmp.add(v)
    var_list = list(tmp)

    return var_list
        
def shuffle_data(*X):
    for x in X:
        N = x.shape[0]
        idx = np.arange(N)
        break
    
    np.random.shuffle(idx)  
    rt = [x[idx] for x in X]
    return rt


def normal_logpdf(x,loc,scale):
    return -0.5*tf.square((x-loc)/scale) - tf.log(scale*tf.sqrt(2.*np.pi))



def concat_cond_data(x,c,one_hot=False,dim=10,conv=False):
    #print('concat check shape',x.shape,c.shape)
    if one_hot:
        c = one_hot_encoder(c,dim)
    
    if isinstance(x,tf.Tensor):
        if conv:
            x_shape = x.get_shape()           
            c = tf.reshape(c,[c.shape[0].value,1,1,-1]) 
            c_shape = c.get_shape()
            #print('x shape',x_shape,'c shape',c_shape)
            cdata = tf.concat([x,c*tf.ones([x_shape[0],x_shape[1],x_shape[2],c_shape[3]])],axis=3)
            #np.concatenate([x,cond*np.ones((x.shape[0],x.shape[1],x.shape[2],cond.shape[3]))],axis=3)
        else: 
            cdata = tf.concat([x,c],axis=1) #np.concatenate([x,cond],axis=1)
    else:
        if conv:
            c = c.reshape((c.shape[0],1,1,-1))
            #print('x shape',x.shape,'c shape',c.shape)
            cdata = np.concatenate([x,c*np.ones((x.shape[0],x.shape[1],x.shape[2],c.shape[3]))],axis=3)
        else: 
            cdata = np.concatenate([x,c],axis=1)

    return cdata

def shuffle_batches(X,batch_size,bc_ids):
    n = len(bc_ids)
    y = np.arange(X.shape[0])
    y_ = np.arange(X.shape[0])
    for i,b in enumerate(bc_ids):
        #print(i,b)
        ids = (y_ < (i+1)*batch_size)&(y_ >= i*batch_size)
        #print(ids)
        y[ids] = y[ids]%batch_size + b * batch_size
    #print(y)

    return X[y]

def condition_mean(x,c):
    c_num = tf.reduce_sum(c,axis=0)+1 #smooth for zeros

    return tf.reduce_sum(x,axis=0)/c_num


def load_cifar10(path=None):

    """Loads CIFAR10 dataset.
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    if path is not None:
        dirname = path + dirname

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
        y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(dirname, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def reinitialize_scope(scope,sess):
    if isinstance(scope,str):
        scope = [scope]
    tmp = []
    for s in scope:
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=s):
            #print('reinit',v)
            tmp.append(v)
    print('reinit var list with length {} in scope {}'.format(len(tmp), scope))
    tf.variables_initializer(tmp).run(session=sess)
    return
        