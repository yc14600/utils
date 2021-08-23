from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf
from scipy import linalg

from .data_util import extract_inception_feature,load_inception_net
from base_models.classifier import Classifier
from base_models.vae import VAE



class FID_Evaluator(object):
    def __init__(self,x_dim,y_dim,net_shape,batch_size,conv=False,sess=None,ac_fn=tf.nn.relu,\
                batch_norm=False,learning_rate=0.001,op_type='adam',decay=None,clip=None,reg=None,
                epoch=100,print_e=20,feature_type='classifier',d_net_shape=None,ipath=None,*args,**kargs):

        self.feature_type = feature_type
        if sess is None:
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        else:
            self.sess = sess

        if net_shape is None:
            if feature_type == 'inception':
                self.feature_model = load_inception_net(ipath)
            else:
                self.feature_model = None

        elif feature_type == 'classifier':
            self.feature_model = Classifier(x_dim,y_dim,net_shape,batch_size,sess=self.sess,epochs=epoch,conv=conv,ac_fn=ac_fn,batch_norm=batch_norm,\
                                            learning_rate=learning_rate,op_type=op_type,decay=decay,clip=clip,reg=reg)

        elif feature_type == 'VAE':
            self.feature_model = VAE(x_dim,y_dim,batch_size,net_shape,d_net_shape,sess=self.sess,epochs=epoch,print_e=print_e,\
                                    learning_rate=learning_rate,conv=conv,reg=reg)

       


        return

    '''
    def config_train(self,learning_rate=0.0002,op_type='adam',beta1=0.5,decay=None,clip=None,*args,**kargs):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_ph,logits=self.d_H[-1])) 
        train,var_list,opt = GAN.config_train(self.loss,scope='discriminator',learning_rate=learning_rate,op_type=op_type,beta1=beta1,decay=decay,clip=clip)

        return train,var_list,opt
    '''

    def train_net(self,X,Y=None,warm_start=False):

        with self.sess.as_default():
            if not warm_start:
                tf.global_variables_initializer().run()
            if self.feature_type == 'classifier':
                self.feature_model.fit(X,Y)
            elif self.feature_type == 'VAE':
                self.feature_model.train(X)


    def get_activations(self,x):
        if self.feature_type == 'classifier':
            z = self.feature_model.extract_feature(x)
        elif self.feature_type == 'VAE':
            z = self.feature_model.encode(x)
        elif self.feature_type == 'inception':
            z = extract_inception_feature(x,self.feature_model)

        return z


    def calc_stats(self,z):
        mu = np.mean(z, axis=0)
        sigma = np.cov(z, rowvar=False)

        return mu,sigma


    def calc_fid(self,mu_a,sigma_a,mu_b,sigma_b):
        eps = 1e-6
        diff = mu_a - mu_b   
        covmean, _ = linalg.sqrtm(sigma_a.dot(sigma_b), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma_a.shape[0]) * eps
            covmean = linalg.sqrtm((sigma_a + offset).dot(sigma_b + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * tr_covmean


    def score(self,x_a,x_b,extractf=True):
        if extractf:
            x_a = self.get_activations(x_a)
            x_b = self.get_activations(x_b)

        mu_a,sigma_a = self.calc_stats(x_a)
        mu_b,sigma_b = self.calc_stats(x_b)

        return self.calc_fid(mu_a,sigma_a,mu_b,sigma_b)




