import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import os

from scipy.stats import multivariate_normal, norm, uniform
from utils.train_util import gen_class_split_data,one_hot_encoder,shuffle_data
from utils.data_util import *
from base_models.gans import fGAN



def str2bool(x):
    if x.lower() == 'false':
        return False
    else:
        return True

def str2ilist(s):   
    s = s[1:-1]
    s = s.split(',')
    l = [int(si) for si in s]
    return l

def str2flist(s):   
    s = s[1:-1]
    s = s.split(',')
    l = [float(si) for si in s]
    return l

def normalize(x):
    s = np.sum(x)
    y = [xi/s for xi in x]
    return y


def get_dists(d_dim,nu_mean,nu_std,de_mean,de_std,nu_pi=[],de_pi=[],dist_type='Normal'):
    
    nu_dist = gen_dist(d_dim,nu_mean,nu_std,nu_pi,dist_type)
    de_dist = gen_dist(d_dim,de_mean,de_std,de_pi,dist_type)
  
    return nu_dist, de_dist


def gen_dist(d_dim,par1,par2,pi=[],dist_type='Normal'):
    if dist_type == 'Normal':
        if len(pi) == 0:
            dist = DiagGaussian(par1,par2,d_dim)
        else:
            dist = MixDiagGaussian(par1,par2,pi,d_dim)

    elif dist_type == 'Uniform':
        dist = Uniform(par1,par2,d_dim)
    
    return dist


def get_dist_list(d_dim,locs,scales,pi=[]):
    dl = []
    if len(pi) == 0:
        for m,s in zip(locs,scales):
            dl.append(gen_dist(d_dim,m,s))
    else:
        for m,s,p in zip(locs,scales,pi):
            dl.append(gen_dist(d_dim,m,s,p))

    return dl

def get_samples(sample_size,nu_dist,de_dist,de_sample_size=None):
    
    nu_samples = nu_dist.sample(sample_size)
    if de_sample_size is None:
        de_samples = de_dist.sample(sample_size) 
    else:
        de_samples = de_dist.sample(de_sample_size)
    return nu_samples, de_samples


# In[12]:


def gen_samples(sample_size,d_dim,loc,scale,dist_type='Normal'):
    if dist_type == 'Normal':
        samples = np.random.normal(loc=loc,scale=scale,size=(sample_size,d_dim)).astype(np.float32)
    elif dist_type == 'Uniform':
        samples = np.random.uniform(low=loc,high=scale,size=(sample_size,d_dim)).astype(np.float32)
    return samples


def Gaussian_KL(dist_a, dist_b,dim):
    return dim * (np.log(dist_b.scale/dist_a.scale)+(np.square(dist_a.scale)+np.square(dist_a.loc-dist_b.loc))/(2*np.square(dist_b.scale))-0.5)


class DiagGaussian(object):
    def __init__(self,mean,std,dim):
        self.loc = mean
        self.scale = std
        self.dim = dim
        if dim > 1:
            self.dist = multivariate_normal(mean=np.ones(dim)*mean,cov=np.ones(dim)*std)
        else:
            self.dist = norm(loc=mean,scale=std)

    def log_prob(self,x):
        return self.dist.logpdf(x)

    def prob(self,x):
        return self.dist.pdf(x)

    def sample(self,size=1):
        return gen_samples(size,self.dim,self.loc,self.scale,dist_type='Normal')


class Uniform(DiagGaussian):
    def __init__(self,lo,ho,dim):
        self.lo = lo
        self.ho = ho
        self.dim = dim
        self.dist = uniform(loc=lo,scale=ho-lo)
    
    def sample(self,size=1):
        return gen_samples(size,self.dim,self.lo,self.ho,dist_type='Uniform')

    def log_prob(self,x):
        return np.sum(self.dist.logpdf(x),axis=1)

    def prob(self,x):
        return np.prod(self.dist.pdf(x),axis=1)



def config_result_path(rpath):
    if rpath[-1] != '/':
        rpath = rpath+'/'

    try:
        os.makedirs(rpath)
    except FileExistsError: #for parallel testing
        return rpath
    
    return rpath


def gen_toygaussian_task_samples(t,sample_size,test_sample_size,args,nu_dists,de_dists):
    nu_samples,de_samples,samples_c = [],[],[]
    if args.validation:
        t_nu_samples,t_de_samples,t_samples_c = [],[],[]
    else:
        t_nu_samples,t_de_samples,t_samples_c = None, None,None
    for c in range(t+1):
        sp_c = (np.ones(sample_size)*c).astype(np.int)
        sp_c = one_hot_encoder(sp_c,args.T)
        #print('check c',sp_c[:5])
        samples_c.append(sp_c)
           
        c_nu_samples,c_de_samples = get_samples(sample_size,nu_dists[c],de_dists[c])
        nu_samples.append(c_nu_samples)
        de_samples.append(c_de_samples)
    
        if args.validation:
            t_sp_c = (np.ones(test_sample_size)*c).astype(np.int)
            t_sp_c = one_hot_encoder(t_sp_c,args.T)
            t_samples_c.append(t_sp_c)
            tc_nu_samples,tc_de_samples = get_samples(test_sample_size,nu_dists[c],de_dists[c])
            t_nu_samples.append(tc_nu_samples)
            t_de_samples.append(tc_de_samples)

    
    ids = np.arange(sample_size*(t+1))
    np.random.shuffle(ids)                 
    samples_c = np.vstack(samples_c)[ids]
    nu_samples = np.vstack(nu_samples)[ids]
    de_samples = np.vstack(de_samples)[ids]

    if args.validation:
        ids = np.arange(test_sample_size*(t+1))
        np.random.shuffle(ids)
        t_samples_c = np.vstack(t_samples_c)[ids]
        t_nu_samples = np.vstack(t_nu_samples)[ids]
        t_de_samples = np.vstack(t_de_samples)[ids]

    return samples_c,nu_samples,de_samples,t_samples_c,t_nu_samples,t_de_samples

# In[13]:
def update_toygaussian_dists(args,t,ori_nu_means,ori_nu_stds,ori_nu_dists,nu_means,nu_stds,de_means,de_stds):
    if args.delta_par == 0. :
        delta_par =  args.delta_list[t+1] #np.random.uniform(-0.5,0.5)
    else:
        delta_par = args.delta_par
    #print('delta par',delta_par)

    if args.continual_ratio:
        nu_means = [mean + delta_par for mean in nu_means]
        nu_stds = [std - delta_par for std in nu_stds] if args.scale_shrink else [std + delta_par for std in nu_stds]
    nu_means.append(ori_nu_means[t+1])
    nu_stds.append(ori_nu_stds[t+1])
    #print('check nu parms',nu_means,nu_stds)

    de_means.append(nu_means[-1])
    de_stds.append(nu_stds[-1])
    de_means = [mean + delta_par for mean in de_means]
    de_stds = [std - delta_par for std in de_stds] if args.scale_shrink else [std + delta_par for std in de_stds]
    #print('check de parms',de_means,de_stds)

    nu_dists = get_dist_list(args.d_dim,nu_means,nu_stds)
    ori_nu_dists.append(nu_dists[-1])    

    de_dists = get_dist_list(args.d_dim,de_means,de_stds)

    return nu_means,nu_stds,nu_dists,de_means,de_stds,de_dists


def load_task_data(t,dpath,x_shape,c_shape,sample_size,test_sample_size):
    fpath = dpath+'task'+str(t)+'/'
    tot_samples = extract_data(fpath+'samples.gz',x_shape)
    tot_labels = extract_labels(fpath+'labels.gz',c_shape,dtype=np.float32)
    class_size = int(tot_samples.shape[0]/(t+1))
    print('tot size',tot_samples.shape,tot_labels.shape,'class size',class_size)
    test_samples, samples, test_labels, labels = [],[],[],[]
    for i in range(t+1):
        samples.append(tot_samples[i*class_size:i*class_size+sample_size])
        labels.append(tot_labels[i*class_size:i*class_size+sample_size])
        if test_sample_size > 0:
            test_samples.append(tot_samples[(i+1)*class_size-test_sample_size:(i+1)*class_size])
            test_labels.append(tot_labels[(i+1)*class_size-test_sample_size:(i+1)*class_size])
    
    samples = np.vstack(samples)
    labels = np.concatenate(labels)
    samples[samples<0.1]=0.
    samples[samples>0.9]=1.
    if test_sample_size > 0:
        test_samples = np.vstack(test_samples)    
        test_labels = np.concatenate(test_labels)
        test_samples[test_samples<0.1]=0.
        test_samples[test_samples>0.9]=1.
    #print('load data check shape',labels.shape,samples.shape,test_labels.shape,test_samples.shape)
    
    
    return labels,samples,test_labels,test_samples

def load_image_data(t,dpath,x_shape,sample_size,test_sample_size):
    tot_samples = extract_data(dpath+'samples.gz',x_shape)
    class_size = int(tot_samples.shape[0]/t)
    print('tot size',tot_samples.shape,'class size',class_size)
    test_samples, samples = [],[]
    for i in range(t):
        samples.append(tot_samples[i*class_size:i*class_size+sample_size])
        
        if test_sample_size > 0:
            test_samples.append(tot_samples[(i+1)*class_size-test_sample_size:(i+1)*class_size])
    
    samples = np.vstack(samples)
    samples[samples<0.1]=0.
    samples[samples>0.9]=1.
    if test_sample_size > 0:
        test_samples = np.vstack(test_samples)    
        test_samples[test_samples<0.1]=0.
        test_samples[test_samples>0.9]=1.    
    
    return samples,test_samples


def gen_task_samples(t,sample_size,test_sample_size,dpath,c_dim,ori_X,ori_Y,ori_test_X=None,ori_test_Y=None,\
                        model_type='continual',tcs=[],shuffled=False):

    nu_samples,de_samples,samples_c = [],[],[]
    test_nu_samples,test_de_samples,test_samples_c = [],[],[]
    print('model type',model_type)
    if model_type == 'bestmodel':
        
        for i in range(t+1):
            nu_samples_i,samples_c_i,test_nu_samples_i,test_samples_c_i = gen_class_split_data(0,sample_size,test_sample_size,ori_X,ori_Y,ori_test_X,ori_test_Y,[i],one_hot=False,C=1)            
            samples_c_i = one_hot_encoder(samples_c_i,c_dim)
            test_samples_c_i = one_hot_encoder(test_samples_c_i,c_dim)
            de_samples_i = shuffle_data(nu_samples_i) #same class but different samples
            test_de_samples_i = shuffle_data(test_nu_samples_i)

            nu_samples.append(nu_samples_i)
            de_samples.append(de_samples_i)
            samples_c.append(samples_c_i)
            test_nu_samples.append(test_nu_samples_i)
            test_de_samples.append(test_de_samples_i)
            test_samples_c.append(test_samples_c_i)

        nu_samples = np.vstack(nu_samples)
        de_samples = np.vstack(de_samples)
        samples_c = np.vstack(samples_c)
        test_nu_samples = np.vstack(test_nu_samples)
        test_de_samples = np.vstack(test_de_samples)
        test_samples_c = np.vstack(test_samples_c)

    elif model_type == 'bestdata': #training data are original data for all tasks
        for i in range(t+1):
            nu_samples_i,samples_c_i,test_nu_samples_i,test_samples_c_i = gen_class_split_data(0,sample_size,test_sample_size,ori_X,ori_Y,ori_test_X,ori_test_Y,[i],one_hot=False,C=1)            
            samples_c_i = one_hot_encoder(samples_c_i,c_dim)
            test_samples_c_i = one_hot_encoder(test_samples_c_i,c_dim)

            nu_samples.append(nu_samples_i)
            samples_c.append(samples_c_i)
            test_nu_samples.append(test_nu_samples_i)
            test_samples_c.append(test_samples_c_i)
        
        nu_samples = np.vstack(nu_samples)
        samples_c = np.vstack(samples_c)
        test_nu_samples = np.vstack(test_nu_samples)
        test_samples_c = np.vstack(test_samples_c)

        x_shape,c_shape = list(nu_samples.shape[1:]),list(samples_c.shape[1:])
        print('x shape',x_shape,'c shape',c_shape)
        d_samples_c, de_samples, test_d_samples_c, test_de_samples = load_task_data(t,dpath,x_shape,c_shape,sample_size,test_sample_size)
        if np.sum(samples_c!=d_samples_c)>0 or np.sum(test_samples_c!=test_d_samples_c)>0  :
            assert('label not aligned!')

    elif model_type == 'taskratio':
        nu_samples,samples_c,test_nu_samples,test_samples_c = gen_class_split_data(0,sample_size,test_sample_size,ori_X,ori_Y,ori_test_X,ori_test_Y,[t],one_hot=False,C=1)            
        
        de_samples,d_samples_c,test_de_samples,test_d_samples_c = gen_class_split_data(0,sample_size,test_sample_size,ori_X,ori_Y,ori_test_X,ori_test_Y,[t+1],one_hot=False,C=1)            

    elif model_type == 'taskratio_pw':
        nu_samples,samples_c,test_nu_samples,test_samples_c = gen_class_split_data(0,sample_size,test_sample_size,ori_X,ori_Y,ori_test_X,ori_test_Y,[tcs[t][0]],one_hot=False,C=1)            
        
        de_samples,d_samples_c,test_de_samples,test_d_samples_c = gen_class_split_data(0,sample_size,test_sample_size,ori_X,ori_Y,ori_test_X,ori_test_Y,[tcs[t][1]],one_hot=False,C=1)            


    elif model_type == 'taskratio_cond':
        samples_c,test_samples_c = [],[]
        nu_samples,samples_c_t,test_nu_samples,test_samples_c_t = gen_class_split_data(0,sample_size,test_sample_size,ori_X,ori_Y,ori_test_X,ori_test_Y,[t],one_hot=False,C=1)            
        de_samples,d_samples_c,test_de_samples,test_d_samples_c = gen_class_split_data(0,sample_size,test_sample_size,ori_X,ori_Y,ori_test_X,ori_test_Y,[t+1],one_hot=False,C=1)            
        for i in range(t+1):
            c_i = np.ones_like(samples_c_t)*i
            tc_i = np.ones_like(test_samples_c_t)*i
            samples_c.append(c_i)
            test_samples_c.append(tc_i)
        if t > 0:        
            nu_samples = np.repeat(nu_samples,t+1,axis=0)
            de_samples = np.repeat(de_samples,t+1,axis=0)
            test_nu_samples = np.repeat(test_nu_samples,t+1,axis=0)
            test_de_samples = np.repeat(test_de_samples,t+1,axis=0)

        samples_c = np.concatenate(samples_c)
        test_samples_c = np.concatenate(test_samples_c)
        samples_c = one_hot_encoder(samples_c,c_dim)
        test_samples_c = one_hot_encoder(test_samples_c,c_dim)
        
    elif model_type == 'single':
        print('single',t,'sample_size',sample_size)
        nu_samples,test_nu_samples = ori_X[:sample_size*t],ori_test_X[:test_sample_size*t]
        x_shape = list(nu_samples.shape[1:])
        if shuffled:
            de_samples = extract_data(dpath,x_shape)
            de_samples,test_de_samples = de_samples[:sample_size*t],de_samples[-test_sample_size*t:]
        else:
            de_samples,test_de_samples = load_image_data(t,dpath,x_shape,sample_size,test_sample_size)
        samples_c, test_samples_c = None, None

    elif model_type == 'splitclass': #nu samples are half classes
        for i in range(int(t/2)):
            nu_samples_i,samples_c_i,test_nu_samples_i,test_samples_c_i = gen_class_split_data(0,sample_size,test_sample_size,ori_X,ori_Y,ori_test_X,ori_test_Y,[i],one_hot=False,C=1)            
            samples_c_i = one_hot_encoder(samples_c_i,c_dim)
            test_samples_c_i = one_hot_encoder(test_samples_c_i,c_dim)

            nu_samples.append(nu_samples_i)
            samples_c.append(samples_c_i)
            test_nu_samples.append(test_nu_samples_i)
            test_samples_c.append(test_samples_c_i)
        
        nu_samples = np.vstack(nu_samples)
        samples_c = np.vstack(samples_c)
        test_nu_samples = np.vstack(test_nu_samples)
        test_samples_c = np.vstack(test_samples_c)
        dsize = int(sample_size*t/2)
        dtsize = int(test_sample_size*t/2)
        de_samples,test_de_samples = ori_X[:dsize],ori_test_X[:dtsize]
    
    elif model_type == 'splitsize': #nu samples are all classes half samples
        de_samples,test_de_samples = ori_X[:sample_size*t],ori_test_X[:test_sample_size*t]
        half_size = int(de_samples.shape[0]/2)
        half_size_t = int(test_de_samples.shape[0]/2)

        nu_samples,test_nu_samples = de_samples[:half_size],test_de_samples[:half_size_t]
        samples_c, test_samples_c = None,None
    else:
        nu_samples,samples_c,test_nu_samples,test_samples_c = gen_class_split_data(0,sample_size,test_sample_size,ori_X,ori_Y,ori_test_X,ori_test_Y,[t],one_hot=False,C=1)
        samples_c = one_hot_encoder(samples_c,c_dim)
        if test_samples_c is not None:
            test_samples_c = one_hot_encoder(test_samples_c,c_dim)
        x_shape,c_shape = list(nu_samples.shape[1:]),list(samples_c.shape[1:])
        print('x shape',x_shape,'c shape',c_shape)
        if t > 0:
            #x_shape[0] = (sample_size + test_sample_size)*t
            #c_shape[0] = (sample_size + test_sample_size)*t
            prev_Y, prev_X,prev_test_Y,prev_test_X = load_task_data(t-1,dpath,x_shape,c_shape,sample_size,test_sample_size)
            print('px shape',prev_X.shape,'py shape',prev_Y.shape)
            nu_samples = np.vstack([prev_X,nu_samples])
            samples_c = np.vstack([prev_Y,samples_c])
            if test_sample_size > 0:
                test_nu_samples = np.vstack([prev_test_X,test_nu_samples])
                test_samples_c = np.vstack([prev_test_Y,test_samples_c])
            
        #x_shape[0] = (sample_size + test_sample_size)*(t+1)
        #c_shape[0] = (sample_size + test_sample_size)*(t+1)
        d_samples_c, de_samples, test_d_samples_c, test_de_samples = load_task_data(t,dpath,x_shape,c_shape,sample_size,test_sample_size)
        if np.sum(samples_c!=d_samples_c)>0 or np.sum(test_samples_c!=test_d_samples_c)>0  :
            assert('label not aligned!')


    #print('gen task data check shape',de_samples.shape,samples_c.shape)    

    ids = np.arange(nu_samples.shape[0])
    np.random.shuffle(ids)
    if samples_c is not None:                 
        samples_c = samples_c[ids]
    nu_samples = nu_samples[ids]
    de_samples = de_samples[ids]
    '''
    test_samples_c = samples_c[-(t+1)*test_sample_size:]
    samples_c = samples_c[:(t+1)*sample_size]
    test_nu_samples = nu_samples[-(t+1)*test_sample_size:]
    nu_samples = nu_samples[:(t+1)*sample_size]
    test_de_samples = de_samples[-(t+1)*test_sample_size:]
    de_samples = de_samples[:(t+1)*sample_size]
    '''
    if test_sample_size >  0:
        ids = np.arange(test_nu_samples.shape[0])
        np.random.shuffle(ids)
        if test_samples_c is not None:                 
            test_samples_c = test_samples_c[ids]
        test_nu_samples = test_nu_samples[ids]
        test_de_samples = test_de_samples[ids]
    print('check shape',nu_samples.shape,de_samples.shape)
    return samples_c,nu_samples,de_samples,test_samples_c,test_nu_samples,test_de_samples


def calc_divgenerce(divergence,samples_ratio,samples_c=None,logr=True):
    
    ds = []
    #print('calc div start check shape',[r.shape for r in samples_log_ratio])
    for r in samples_ratio:
        if samples_c is not None:
            c_num = np.sum(samples_c,axis=0)
            if len(r.shape) < 2:
                r = r.reshape(-1,1)       

            if divergence == 'rv_KL':            
                log_r = r if logr else np.log(r)
                d = np.sum(-log_r*samples_c,axis=0)/c_num   

            else:
                r = np.exp(r) if logr else r
                f = fGAN.get_f(divergence)
                d = np.sum(f(r)*samples_c,axis=0)/c_num
        else:
            if divergence == 'rv_KL':            
                log_r = r if logr else np.log(r)
                d = np.mean(-log_r)
            elif divergence == 'KL' and logr:
                d = np.mean(np.exp(r)*r)
            else:
                r = np.exp(r) if logr else r
                f = fGAN.get_f(divergence)
                d = np.mean(f(r))

        ds.append(d)
    return ds

