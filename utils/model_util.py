from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import edward as ed
#import matplotlib.pyplot as plt
#import seaborn as sb
import numpy as np
import scipy as sp
import collections
import six
import os
path = os.getcwd()
import sys
sys.path.append(path+'/../')

from tensorflow.keras.initializers import Initializer
from utils.train_util import get_next_batch
from edward.models import Normal,OneHotCategorical,MultivariateNormalTriL
from hsvi.methods.svgd import SVGD

class ReprmNormal:
    def __init__(self,loc,logvar):
        self.loc = loc
        self.logvar = logvar
        self.scale = tf.exp(0.5*logvar)
        self._value = self.sample()

    def log_prob(self,x):
        #if isinstance(x,ReprmNormal):
        #    x = x.sample()
        const = -tf.log(2.*np.pi)
        log_p = 0.5*(const - self.logvar - tf.square(self.loc-x)*tf.exp(-self.logvar))
        return log_p

    def sample(self,shape=()):
        e = Normal(loc=tf.zeros_like(self.loc),scale=tf.ones_like(self.logvar))
        es = e.sample(shape)
        return self.loc+es*self.scale

    def value(self):
        return self._value
    
    def kl_divergence(self,p):
        ratio = tf.exp(self.logvar)/tf.square(p.scale)
        return 0.5*(tf.square((self.loc-p.loc)/(p.scale))+ratio-tf.log(ratio)-1.)
    

def construct_scale_tril(K):
    mask = np.arange(K*K).reshape(K,K)
    ri = np.floor(mask/K)
    ci = mask - ri*K
    mask = (ci<=ri).astype(np.float32)
    tril_s = tf.Variable(tf.random_normal([K,K],mean=0.,stddev=0.001)+tf.eye(K)*0.01)
    tril_s = tf.multiply(tril_s,mask) 

    return tril_s


def define_Normal_trans_parm(l,qw,name='W'):
    AW = tf.Variable(tf.ones_like(qw),name='layer'+str(l)+'_A'+name)
    BW = tf.Variable(tf.zeros_like(qw),name='layer'+str(l)+'_B'+name)
    OW = tf.Variable(tf.ones_like(qw)*-3.,name='layer'+str(l)+'_O'+name)
    pW = Normal(loc=tf.zeros_like(qw),scale=tf.ones_like(qw))
    tW_v = tf.exp(2.*OW) + tf.ones_like(qw)
    tW = Normal(loc=AW*pW+BW, scale=tf.sqrt(tW_v))
    return tW,pW,AW,OW,BW


def gen_Normal_trans(mu,sigma,qw_parms):
    qw_trans_v = tf.square(sigma*qw_parms[0])+ tf.exp(2.*qw_parms[1])
    qw_trans = Normal(loc=qw_parms[0]*mu+qw_parms[2],scale=tf.sqrt(qw_trans_v))
    return qw_trans


# to be completed with more distirbutions and prior setting
def gen_posterior_conf(qW_list,distribution='gaussian',prior='uniform'):
    cfg = {}
    if distribution == 'gaussian':
        if prior == 'uniform':
            for qw in qW_list:
                pw = Normal(loc=tf.zeros_like(qw),scale=tf.ones_like(qw))
                cfg[pw] = qw
    return cfg


def update_variable_tables(pW,qW,sess,task_var_cfg,trans_var_cfg=None,transition_parm=None,transition_parm_ng=None,qW_prior=None):
    # update weights prior
    mu = sess.run(pW.loc)
    sigma = sess.run(pW.scale)
    npw = Normal(loc=mu, scale=sigma)

    # update weights transition
    if transition_parm is not None:
        qw_trans = gen_Normal_trans(mu,sigma,transition_parm[qW])
        task_var_cfg[qw_trans] = qW
        if trans_var_cfg is not None:
            trans_var_cfg[npw] = qw_trans
        if transition_parm_ng is not None:
            transition_parm_ng[qw_trans] = transition_parm[qW]
        if qW_prior is not None:
            qW_prior.append(qw_trans)
    else:
        task_var_cfg[npw] = qW
        if qW_prior is not None:
            qW_prior.append(qW)
    

def define_dense_layer(l,d1,d2,initialization=None,reg=None):
    w_name = 'dense_layer'+str(l)+'_weights'
    b_name = 'dense_layer'+str(l)+'_bias'

    if reg=='l2' :
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01) 
    elif reg=='l1':
        regularizer = tf.contrib.layers.l1_regularizer(scale=0.01) 
    else:
        regularizer = None

    if initialization is None:
        w = tf.get_variable(name=w_name,initializer=tf.random_normal([d1,d2],stddev=1./np.sqrt(d1)),regularizer=regularizer)
        b = tf.get_variable(name=b_name,initializer=tf.zeros([d2]),regularizer=regularizer)
    else:
        W0 = initialization['w']
        if isinstance(W0, collections.Iterable):
            W0 = W0[l]
        
        B0 = initialization['b']
        if isinstance(B0, collections.Iterable):
            B0 = B0[l]

        if isinstance(W0,Initializer):
            w = tf.get_variable(name=w_name,shape=[d1,d2],initializer=W0,regularizer=regularizer)
        else:
            w = tf.get_variable(name=w_name,initializer=W0,regularizer=regularizer)

        if isinstance(B0,Initializer):
            b = tf.get_variable(name=b_name,shape=[d2],initializer=B0,regularizer=regularizer)
        else:
            b = tf.get_variable(name=b_name,initializer=B0,regularizer=regularizer)

    return w, b 


def linear(x,w,b):
    #print('linear func check',x.shape,w.shape,b.shape)
    return tf.add(tf.matmul(x,w),b)


def build_dense_layer(x,l,d1,d2,initialization=None,ac_fn=tf.nn.relu,batch_norm=False,training=None,scope=None,reg=None):
    print('dense layer',l,'batch norm',batch_norm,'activation',ac_fn,'regularizer',reg)
    w,b = define_dense_layer(l,d1,d2,initialization,reg)
    h = linear(x,w,b)
    if batch_norm:
        h = tf.contrib.layers.batch_norm(h,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=training,scope=scope+'/bn/layer'+str(l))
    if ac_fn is not None:
        h = ac_fn(h)
    
    return w,b,h
    

def restore_dense_layer(x,l,w,b,ac_fn=tf.nn.relu,batch_norm=False,training=None,scope=''):
    h = linear(x,w,b)
    if batch_norm:
        h = tf.contrib.layers.batch_norm(h,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=training,scope=scope+'/bn/layer'+str(l))

    if ac_fn is not None:
        h = ac_fn(linear(x,w,b))
    return h


def define_gaussian_dense_layer(l,gaussian_type,d1,d2,initialization=None,share='isotropic'):
    tril_s = None
    w_m_init, w_s_init = None,None
    b_m_init, b_s_init = None,None
    w_m_name = 'layer'+str(l)+'w_mean'
    w_s_name = 'layer'+str(l)+'w_var'
    b_m_name = 'layer'+str(l)+'b_mean'
    b_s_name = 'layer'+str(l)+'b_var'
    print('layer ',l,d1,d2)
    if initialization is not None:
        w_m_init = initialization.get('w_m',None)
        w_s_init = initialization.get('w_s',None)
        b_m_init = initialization.get('b_m',None)
        b_s_init = initialization.get('b_s',None)

    if w_m_init is None:
        w_loc_var = tf.Variable(tf.random_normal([d1,d2],stddev=0.001),name=w_m_name)
    else:
        w_loc_var = tf.Variable(w_m_init,name=w_m_name)
    
    if b_m_init is None:
        b_loc_var = tf.Variable(tf.random_normal([d2],stddev=0.001),name=b_m_name)
    else:
        b_loc_var = tf.Variable(b_m_init,name=b_m_name)

    if w_s_init is None:
        if share == 'variance':
            w_s_var = tf.Variable(1./np.sqrt(d1),name=w_s_name)
        elif share == 'column_covariance':                 
            w_s_var = construct_scale_tril(d2) 
        elif share == 'row_covariance':
            w_s_var = construct_scale_tril(d1)
        elif share == 'isotropic':
            w_s_var = tf.Variable(tf.ones([d1,d2])*-3.,name=w_s_name)
    else:
        if share == 'isotropic':
            w_s_var = tf.Variable(tf.ones([d1,d2])*w_s_init,name=w_s_name)
        else:
            w_s_var = tf.Variable(w_s_init,name=w_s_name)
    
    if b_s_init is None:
        if share == 'variance':
            b_s_var = tf.Variable(1./np.sqrt(d1),name=b_s_name)
        elif share == 'column_covariance' or share == 'row_covariance':                 
            b_s_var = tf.Variable(tf.ones([d2])*1./np.sqrt(d2),name=b_s_name)
        elif share == 'isotropic':
            b_s_var = tf.Variable(tf.ones([d2])*-3.,name=b_s_name)    
    else:
        if share == 'isotropic':
            b_s_var = tf.Variable(tf.ones([d2])*b_s_init,name=b_s_name)
        else:
            b_s_var = tf.Variable(b_s_init,name=b_s_name)      
        
    if share == 'variance':   
        w = Normal(loc=w_loc_var,scale=tf.nn.softplus(w_s_var))
        b = Normal(loc=b_loc_var,scale=tf.nn.softplus(b_s_var))
    
    elif share == 'column_covariance':                  
        w = MultivariateNormalTriL(loc=w_loc_var,scale_tril=w_s_var)
        b = Normal(loc=b_loc_var,scale=tf.nn.softplus(b_s_var))
    
    elif share == 'row_covariance':                             
        w = MultivariateNormalTriL(loc=tf.transpose(w_loc_var),scale_tril=w_s_var)
        b = Normal(loc=b_loc_var,scale=tf.nn.softplus(b_s_var))      

    elif share == 'isotropic':        
        if gaussian_type == 'common':           
            w = Normal(loc=w_loc_var,scale=tf.nn.softplus(w_s_var))
            b = Normal(loc=b_loc_var,scale=tf.nn.softplus(b_s_var))
        elif gaussian_type == 'logvar':
            w = Normal(loc=w_loc_var,scale=tf.exp(w_s_var))
            b = Normal(loc=b_loc_var,scale=tf.exp(b_s_var))
          
    return w,b,tril_s,{w:[w_loc_var,w_s_var],b:[b_loc_var,b_s_var]}


def define_gaussian_conv_layer(l,filter_shape,initialization=None):
    
    w_mean_name = 'convlayer'+str(l)+'w_mean'
    w_scale_name = 'convlayer'+str(l)+'w_var'

    if initialization is not None:
        m_init = initialization.get('cw_m',None)
        s_init = initialization.get('cw_s',None)
    else:
        m_init,s_init = None, None

    if m_init is None:
        w_loc_var = tf.get_variable(w_mean_name,shape=filter_shape,dtype=tf.float32,initializer=tf.random_normal_initializer(mean=0.,stddev=1e-6))
    else:
        w_loc_var = tf.get_variable(w_mean_name,dtype=tf.float32,initializer=m_init)

    if s_init is None:
        w_s_var = tf.get_variable(w_scale_name,dtype=tf.float32,initializer=tf.ones_like(w_loc_var)*-5.)
    else:
        w_s_var = tf.get_variable(w_scale_name,dtype=tf.float32,initializer=tf.ones_like(w_loc_var)*s_init)

   
    w = Normal(loc=w_loc_var,scale=tf.exp(w_s_var))
      
    return w,{w:[w_loc_var,w_s_var]}


def define_conv_layer(l,filter_shape,initialization=None,deconv=False,reg=None):
    w_name = 'conv_layer_weights'+str(l)
    b_name = 'conv_layer_bias'+str(l)

    b_shape = [filter_shape[-2]] if deconv else [filter_shape[-1]]

    if reg=='l2' :
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1) 
    elif reg=='l1':
        regularizer = tf.contrib.layers.l1_regularizer(scale=0.1) 
    else:
        regularizer = None

    if initialization is None:
        w_var = tf.get_variable(w_name,shape=filter_shape,dtype=tf.float32,initializer=tf.random_normal_initializer(mean=0.,stddev=1e-6),regularizer=regularizer)
        b_var = tf.get_variable(b_name, b_shape, initializer=tf.constant_initializer(0.0),regularizer=regularizer)
    else:
        w0 = initialization['w']
        b0 = initialization['b']
        if isinstance(w0, collections.Iterable):
            w0 = w0[l]
        if isinstance(b0, collections.Iterable):
            b0 = b0[l]

        if isinstance(w0, Initializer):           
            w_var = tf.get_variable(w_name,shape=filter_shape,dtype=tf.float32,initializer=w0,regularizer=regularizer)
        else:
            w_var = tf.get_variable(w_name,dtype=tf.float32,initializer=w0,regularizer=regularizer)

        if isinstance(b0, Initializer):           
            b_var = tf.get_variable(b_name,shape=b_shape,dtype=tf.float32,initializer=b0,regularizer=regularizer)
        else:
            b_var = tf.get_variable(b_name,dtype=tf.float32,initializer=b0,regularizer=regularizer)

    return w_var,b_var


def build_conv_layer(x,l,filter_shape,strides=[1,2,2,1],padding='SAME',initialization=None,deconv=False,output_shape=None,reg=None):
    w, b = define_conv_layer(l,filter_shape,initialization,deconv,reg)
    if deconv:
        h = tf.nn.conv2d_transpose(x,filter=w,output_shape=output_shape,strides=strides,padding=padding)
    else:
        h = tf.nn.conv2d(input=x,filter=w,strides=strides,padding=padding)

    h = tf.reshape(tf.nn.bias_add(h, b), h.get_shape())
    return w,b,h


def build_conv_bn_acfn(x,l,filter_shape,strides=[1,2,2,1],padding='SAME',initialization=None,deconv=False,\
                        output_shape=None,batch_norm=False,ac_fn=tf.nn.relu,training=None,scope=None,reg=None):
    print('conv layer',l,'batch norm',batch_norm,'activation',ac_fn)
    w,b,h = build_conv_layer(x,l,filter_shape,strides,padding,initialization,deconv,output_shape,reg)
    if batch_norm:
        h = tf.contrib.layers.batch_norm(h,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=training,scope=scope+'/bn/convlayer'+str(l))
    h = ac_fn(h)
    return w,b,h


def restore_conv_layer(x,l,w,b,strides=[1,2,2,1],padding='SAME',initialization=None,deconv=False,\
                        output_shape=None,batch_norm=False,ac_fn=tf.nn.relu,training=None,scope=''):
    
    if deconv:
        h = tf.nn.conv2d_transpose(x,filter=w,output_shape=output_shape,strides=strides,padding=padding)
    else:
        h = tf.nn.conv2d(input=x,filter=w,strides=strides,padding=padding)

    h = tf.reshape(tf.nn.bias_add(h, b), h.get_shape())

    if batch_norm:
        h = tf.contrib.layers.batch_norm(h,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=training,scope=scope+'/bn/convlayer'+str(l))
    h = ac_fn(h)

    return h


def local_reparam(x,w,parm_var,conv2d=True,strides=[1,2,2,1],padding='SAME'):
    if conv2d:
        h_mu = tf.nn.conv2d(input=x,filter=parm_var[w][0],strides=strides,padding=padding)
        h_v = tf.nn.conv2d(input=tf.square(x),filter=tf.square(tf.nn.softplus(parm_var[w][1])),strides=strides,padding=padding)        
    else:
        h_mu = tf.matmul(x,parm_var[w][0])
        h_v = tf.matmul(tf.square(x),tf.square(tf.nn.softplus(parm_var[w][1])))
    h = Normal(loc=h_mu,scale=tf.sqrt(h_v))
    return h


def build_bayesian_conv_layer(x,l,filter_shape,strides=[1,2,2,1],padding='SAME',initialization=None,num_samples=10,local_rpm=True):
    w,parm_var = define_gaussian_conv_layer(l,filter_shape,initialization) 
    if local_rpm:  
        h = local_reparam(x,w,parm_var,conv2d=True,strides=strides,padding=padding)
        #h = tf.reduce_mean(h.sample(num_samples),axis=0)
    else: 
        h = tf.nn.conv2d(input=x,filter=w,strides=strides,padding=padding)
    return h,w,parm_var


def build_bayesian_conv_bn_acfn(x,l,filter_shape,strides=[1,2,2,1],padding='SAME',initialization=None,num_samples=10,batch_norm=False,local_rpm=False,ac_fn=tf.nn.relu):
    h,w,parm_var = build_bayesian_conv_layer(x,l,filter_shape,strides,padding,initialization,num_samples,local_rpm)
    if batch_norm:
        h = tf.layers.batch_normalization(h)
    #h = tf.contrib.layers.layer_norm(h)
    h = ac_fn(h)
    return h,w,parm_var


def compute_head_output(h,w,b,bayes=False,share='isotropic',num_samples=10,local_rpm=False):
    ew,eb = None, None
    if bayes:
        if local_rpm:
            if share == 'row_covariance':
                tw = tf.transpose(w)
            
            else: 
                tw = w
            #z = h.sample(num_samples)
            #ez = tf.einsum('sbi,ik->sbk',z,tw)#z = local_reparam(h,tw,parm_var,conv2d=False)
            z = tf.matmul(h,tw)
            #eb = b.sample(num_samples)

            #h = OneHotCategorical(logits=ez+tf.expand_dims(eb,1)) 
            h = OneHotCategorical(logits=z+b)                     
             
            
        else:
            ew = w.sample(num_samples)
            eb = b.sample(num_samples)

            if share == 'row_covariance':
                tew = tf.reshape(ew,[-1,ew.shape[2].value,ew.shape[1].value])
                z = tf.einsum('sbi,sij->sbj',h,tew)+tf.expand_dims(eb,1)
                h = OneHotCategorical(logits=z)

            else:                       
                z = tf.einsum('sbi,sij->sbj',h,ew)+tf.expand_dims(eb,1)
                h = OneHotCategorical(logits=z)
    else:
        h = tf.nn.softmax(tf.add(tf.matmul(h,w),b))
    return h,ew,eb


def compute_layer_output(input,w,b,bayes=False,ac_fn=tf.nn.relu,share='isotropic',num_samples=10,head=False,num_heads=1,local_rpm=False,parm_var=None):
    h = input
    ew, eb = None, None
    if head:
        if num_heads == 1:
            h,ew,eb = compute_head_output(h,w,b,bayes,share,num_samples,local_rpm=local_rpm)
        elif num_heads > 1:
            Y = []
            if bayes:
                ew, eb = [], []
                for wi,bi in zip(w,b):
                    hi,ewi,ebi = compute_head_output(h,wi,bi,bayes,share,num_samples,local_rpm=local_rpm)
                    Y.append(hi)
                    ew.append(ewi)
                    eb.append(ebi)
            else:
                for wi,bi in zip(w,b):
                    hi,ewi,ebi = compute_head_output(h,wi,bi,bayes,share,num_samples)
                    Y.append(hi)
            return Y,ew,eb
    # middle layers   
    else:
        if bayes:
            if local_rpm:
                if share == 'row_covariance':
                    tw = tf.transpose(w)
                   
                else:
                    tw = w

                z = local_reparam(h,tw,parm_var,conv2d=False) 
                #ez = z.sample(num_samples)
                #eb = b.sample(num_samples)

                #h = ac_fn(tf.reduce_mean(ez+tf.expand_dims(eb,1),axis=0))
                h = ac_fn(z+b)
                return h,ew,eb
            else:
                ew = w.sample(num_samples)
                eb = b.sample(num_samples)

                if share == 'row_covariance':
                    tew = tf.reshape(ew,[-1,ew.shape[2].value,ew.shape[1].value])
                    z = tf.einsum('sbi,sij->sbj',h,tew)+tf.expand_dims(eb,1)
                    h = ac_fn(z)
                else:
                    z = tf.einsum('sbi,sij->sbj',h,ew)+tf.expand_dims(eb,1)
                    h = ac_fn(z)
        else:
            h = ac_fn(tf.add(tf.matmul(h,w),b))
    return h,ew,eb


def build_nets(net_shape,input,bayes=False,ac_fn=tf.nn.relu,share='isotropic',initialization=None,\
                num_samples=1,gaussian_type='logvar',num_heads=1,dropout=None,local_rpm=False):
        W = []
        B = []
        parm_var = {}
        if bayes and len(input.shape) < 3 and not local_rpm:
            h = tf.expand_dims(input,axis=0)
            h = tf.tile(h,[num_samples,1,1])
            
        else:
            h = input
        H =[]
        TS = {}
        W_samples = []
        B_samples = []
        for l in range(len(net_shape)-1):

            ## define variable
            d1 = net_shape[l]
            d2 = net_shape[l+1]
            
            if bayes:                  
                w,b,tril_s,l_par_var = define_gaussian_dense_layer(l,gaussian_type,d1,d2,initialization,share)
                if 'covariance' in share:
                    TS[w] = tril_s
                parm_var.update(l_par_var)
            # NOT Bayes NN
            else: 
                w, b = define_dense_layer(l,d1,d2,initialization)                         

            ## compute head layer output
            if l == len(net_shape) - 2:
                if num_heads == 1:
                    h,ew,eb = compute_layer_output(h,w,b,bayes,ac_fn,share,num_samples,head=True,num_heads=num_heads,local_rpm=local_rpm)
                elif num_heads > 1:
                    W_last = [w]
                    B_last = [b]                    
                    for k in range(num_heads-1):
                        if bayes:
                            w,b,tril_s,l_par_var = define_gaussian_dense_layer(l,gaussian_type,d1,d2,initialization,share)
                            if 'covariance' in share:
                                TS[w] = tril_s
                            parm_var.update(l_par_var)
                        else:
                            w,b = define_dense_layer(l,d1,d2,initialization) 
                        W_last.append(w)
                        B_last.append(b)

                    Y,eW,eB = compute_layer_output(h,W_last,B_last,bayes,ac_fn,share,num_samples,head=True,num_heads=num_heads,local_rpm=local_rpm)
                    W_list = [W+[wi] for wi in W_last]
                    B_list = [B+[bi] for bi in B_last]
                    H_list = [H+[yi] for yi in Y]
                    W_list_samples = [W_samples+[ew] for ew in eW]
                    B_list_samples = [B_samples+[eb] for eb in eB]
                    return W_list,B_list,H_list,TS,W_list_samples,B_list_samples,parm_var
            ## compute middle hidden units
            else:
                h,ew,eb = compute_layer_output(h,w,b,bayes,ac_fn,share,num_samples,head=False,local_rpm=local_rpm,parm_var=parm_var)
                if dropout is not None and l == len(net_shape) - 3:
                    h = tf.nn.dropout(h,keep_prob=dropout)

            W.append(w)
            B.append(b)
            H.append(h)
            W_samples.append(ew)
            B_samples.append(eb)

        return W,B,H,TS,W_samples,B_samples,parm_var

def forward_nets(W,B,input,ac_fn=tf.nn.tanh,bayes=False,num_samples=1,local_rpm=False,parm_var=None,output_ac=tf.nn.softmax):
    if bayes and not local_rpm:
        h = tf.expand_dims(input,axis=0)
        h = tf.tile(h,[num_samples,1,1])
    else:
        h = input

    if bayes:
        if local_rpm:
            for l in range(len(B)-1):
                z = local_reparam(h,W[l],parm_var)+B[l]
                h = ac_fn(z)
            
            z = local_reparam(h,W[l],parm_var)+B[l]          
        else:
            for l in range(len(B)-1):
                ew = W[l].sample(num_samples)
                eb = B[l].sample(num_samples)
                z = tf.einsum('sbi,sij->sbj',h,ew)+tf.expand_dims(eb,1)
                h = ac_fn(z)
            ew = W[-1].sample(num_samples)
            eb = B[-1].sample(num_samples)
            z = tf.einsum('sbi,sij->sbj',h,ew)+tf.expand_dims(eb,1)
        h = OneHotCategorical(logits=z)
    else:
        
        for l in range(len(B)-1):
            h = forward_dense_layer(h,W[l],B[l],ac_fn) #ac_fn(tf.add(tf.matmul(h,W[l]),B[l]))
        
        h = forward_dense_layer(h,W[-1],B[-1],output_ac)#tf.nn.softmax(tf.add(tf.matmul(h,W[-1]),B[-1]))
        
    return h


def forward_dense_layer(x,w,b,ac_f):
    return ac_f(tf.add(tf.matmul(x,w),b))


def forward_mean_nets(qW,qB,x_ph,ac_fn,sess,share_type='isotropic'):
    mW,mB = [],[]
    for l in range(len(qW)):
        if share_type == 'row_covariance':
            mW.append(sess.run(qW[l].loc).transpose())
        else:
            mW.append(sess.run(qW[l].loc))      
        mB.append(sess.run(qB[l].loc))
    
    my = forward_nets(mW,mB,x_ph,ac_fn=ac_fn)
    return my

def fit_model(num_iter, x_train, y_train,x_ph,y_ph,batch_size,train_step,loss,sess, print_iter=100):
    ii = 0
    for _ in range(num_iter):
            x_batch,y_batch,ii = get_next_batch(x_train,batch_size,ii,labels=y_train)
            feed_dict = {x_ph:x_batch,y_ph:y_batch}
            l,__ = sess.run([loss,train_step], feed_dict=feed_dict)
            if _% print_iter==0:
                print('loss',l)

def predict(x_test,y_test,x_ph,y,batch_size,sess,regression=False):
        n = int(np.ceil(x_test.shape[0]/batch_size))
        r = x_test.shape[0]%batch_size
        correct = 0.
        ii = 0
        result = []
        for i in range(n):
            x_batch,y_batch,ii = get_next_batch(x_test,batch_size,ii,labels=y_test,repeat=True)
            feed_dict = {x_ph:x_batch}
            y_pred_prob = sess.run(y,feed_dict=feed_dict)
            if i == n-1 and r>0:
                y_pred_prob = y_pred_prob[-r:]
                y_batch = y_batch[-r:]
            #print('y pred prob',y_pred_prob.shape,y_test.shape,np.sum(y_test,axis=0))

            if len(y_pred_prob.shape) > 2:
                y_pred_prob = np.mean(y_pred_prob,axis=0)
                
            #if regression:
            result.append(y_pred_prob)
            
            y_pred = np.argmax(y_pred_prob,axis=1)
            correct += np.sum(np.argmax(y_batch,axis=1)==y_pred)
            
        #if regression:
        result = np.vstack(result)        
        acc = correct/y_test.shape[0]
        return acc, result


def LinearRegression(x_ph,in_dim,out_dim,Bayes=True,initialization=None,num_samples=1,logistic=True,noise_std=0.1):
    N = in_dim
    C = out_dim
    if Bayes:
        if initialization is None:
            w_loc_var = tf.Variable(tf.random_normal([N,C],stddev=.1),name='weights_mean')
            b_loc_var = tf.Variable(tf.random_normal([C],stddev=.1),name='bias_mean')
            w_s_var = tf.Variable(tf.ones([N,C])*-3.,name='weights_std')
            b_s_var = tf.Variable(tf.ones([C])*-3.,name='bias_std')
            
        else:
            W0 = initialization['w']
            B0 = initialization['b']
            w_loc_var = tf.Variable(W0,name='weights_mean')
            b_loc_var = tf.Variable(B0,name='bias_mean')
            w_s_var = tf.Variable(tf.ones([N,C])*-3.,name='weights_std')
            b_s_var = tf.Variable(tf.ones([C])*-3.,name='bias_std')

        W = Normal(loc=w_loc_var,scale=tf.exp(w_s_var))
        B = Normal(loc=b_loc_var,scale=tf.exp(b_s_var))
        #eW = tf.reduce_mean(W.sample(num_samples),axis=0)
        if logistic:
            y = OneHotCategorical(logits=tf.matmul(x_ph,W)+B)
        else:
            y = Normal(loc=tf.matmul(x_ph,W)+B,scale=noise_std)
        return W,B,y,{W:[w_loc_var,w_s_var],B:[b_loc_var,b_s_var]}
    else:
        if initialization is None:
            W = tf.Variable(tf.random_normal([N,C],stddev=.1),name='weights')
            B = tf.Variable(tf.random_normal([C],stddev=.1),name='bias')
        else:
            W = tf.Variable(initialization['w'],name='weights')
            B = tf.Variable(initialization['b'],name='bias')
        #eW = tf.reduce_mean(W.sample(num_samples),axis=0)
        if logistic:
            y = tf.nn.softmax(tf.matmul(x_ph,W)+B)
        else:
            y = tf.matmul(x_ph,W)+B
        return W,B,y

def compute_diag_fisher(ll,parm_var):
    fisher = {}
    avg = 1./ll.shape[0].value
    for parm,var in six.iteritems(parm_var):
        fisher[parm] = 0.
        for s in range(ll.shape[0]):
            g = tf.gradients(ll[s],var)            
            fisher[parm] += tf.square(g[0])            
        fisher[parm]*=avg
    return fisher


def test_tasks(t,test_sets,qW,qB,num_heads,x_ph,ac_fn,batch_size,sess,conv_h=None):
    acc_record, pred_probs = [], []
    if conv_h is not None:
        in_x = conv_h
    else:
        in_x = x_ph

    if num_heads > 1:  
        my = []
        for k in range(t+1):
            my.append(forward_mean_nets(qW[k],qB[k],in_x,ac_fn,sess))
    else:
        my = forward_mean_nets(qW,qB,in_x,ac_fn,sess)
        
    for k,ts in enumerate(test_sets):   
        if num_heads > 1:
            acc, y_probs = predict(ts[0],ts[1],x_ph,my[k],batch_size,sess)  
        else: 
            acc, y_probs = predict(ts[0],ts[1],x_ph,my,batch_size,sess)
        print('accuracy',acc)
        acc_record.append(acc)
        pred_probs.append(y_probs)
    print('avg accuracy',np.mean(acc_record))
    return acc_record,pred_probs

def gen_trans_parms(q_list,name,transition_parm={},transition_parm_ng={},task_var_cfg={},trans_var_cfg={}):
    for i,qw in enumerate(q_list):
        tW,pW,AW,OW,BW = define_Normal_trans_parm(i,qw,name)
        transition_parm[qw] = (AW,OW,BW)
        transition_parm_ng[tW] = transition_parm[qw]
        
        task_var_cfg[tW] = qw
        trans_var_cfg[pW] = tW
    return transition_parm,transition_parm_ng,task_var_cfg,trans_var_cfg

def cifar_model(x,batch_size,local_rpm=False,initialization=None):
    conv_W = []
    parm_var_dict = {}
    # first layer conv2d
    filter_shape = [3,3,3,32]
    strides = [1,2,2,1]
    h,w,parm_var = build_bayesian_conv_bn_acfn(x,0,filter_shape,strides=strides,local_rpm=local_rpm,initialization=initialization)
    conv_W.append(w)
    parm_var_dict.update(parm_var)
    print('L1',h.shape)
    # second layer conv2d
    filter_shape = [3,3,32,32]
    strides = [1,1,1,1]
    h,w,parm_var = build_bayesian_conv_bn_acfn(h,1,filter_shape,strides=strides,local_rpm=local_rpm,initialization=initialization)
    conv_W.append(w)
    parm_var_dict.update(parm_var)
    print('L2',h.shape)
    '''
    # pooling
    h = tf.nn.max_pool(value=h,ksize=[1,2,2,1],strides=strides,padding='SAME')
    # thrid layer conv2d
    filter_shape = [3,3,32,64]
    strides = [1,1,1,1]
    h,w,parm_var = build_bayesian_conv_bn_acfn(h,2,filter_shape,strides=strides,local_rp=False)
    conv_W.append(w)
    parm_var_dict.update(parm_var)
    print('L3',h.shape)
    # fourth layer conv2d
    filter_shape = [3,3,64,64]
    strides = [1,1,1,1]
    h,w,parm_var = build_bayesian_conv_bn_acfn(h,3,filter_shape,strides=strides,local_rp=False)
    conv_W.append(w)
    parm_var_dict.update(parm_var)
    print('L4',h.shape)
    '''
    # pooling
    h = tf.nn.max_pool(value=h,ksize=[1,2,2,1],strides=strides,padding='SAME')
    # flatten
    h = tf.reshape(h,shape=[-1,h.shape[1]*h.shape[2]*h.shape[3]])
    print('flatten',h)
    return conv_W,parm_var_dict,h

def forward_conv2d_bn_acfn(x,w,strides,padding,ac_fn=tf.nn.relu,local_rp=False,parm_var=None):
    if local_rp:  
        h = local_reparam(x,w,parm_var,conv2d=True,strides=strides,padding=padding)
        #h = tf.reduce_mean(h.sample(num_samples),axis=0)
    else: 
        h = tf.nn.conv2d(input=x,filter=w,strides=strides,padding=padding)
    #h = tf.layers.batch_normalization(h)
    h = ac_fn(h)
    return h 

def forward_cifar_model(x,W,local_rp=False):

    strides = [1,2,2,1]
    h = forward_conv2d_bn_acfn(x,W[0],strides,padding='SAME')
    strides = [1,1,1,1]
    h = forward_conv2d_bn_acfn(h,W[1],strides,padding='SAME')
    h = tf.nn.max_pool(value=h,ksize=[1,2,2,1],strides=strides,padding='SAME')
    h = tf.reshape(h,shape=[-1,h.shape[1]*h.shape[2]*h.shape[3]])

    return h