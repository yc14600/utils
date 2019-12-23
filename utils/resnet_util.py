
import math
import numpy as np
import tensorflow as tf

from .model_util import *

########## functions are borrowed from https://github.com/facebookresearch/agem ##########

def _conv(x, kernel_size, out_channels, stride, var_list, pad="SAME", scope="conv"):
    """
    Define API for conv operation. This includes kernel declaration and
    conv operation both.
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        #n = kernel_size * kernel_size * out_channels
        n = kernel_size * in_channels
        stdv = 1.0 / math.sqrt(n)
        w = tf.get_variable('kernel', [kernel_size, kernel_size, in_channels, out_channels],
                           tf.float32, 
                           initializer=tf.random_uniform_initializer(-stdv, stdv))
                           #initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))

        # Append the variable to the trainable variables list
        var_list.append(w)

    # Do the convolution operation
    output = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=pad)
    return output


def _fc(x, out_dim, var_list, scope="fc", is_cifar=True):
    """
    Define API for the fully connected layer. This includes both the variable
    declaration and matmul operation.
    """
    in_dim = x.get_shape().as_list()[1]
    stdv = 1.0 / math.sqrt(in_dim)
    with tf.variable_scope(scope):
        # Define the weights and biases for this layer
        w = tf.get_variable('weights', [in_dim, out_dim], tf.float32, 
                initializer=tf.random_uniform_initializer(-stdv, stdv))
                #initializer=tf.truncated_normal_initializer(stddev=0.1))
        if is_cifar:
            b = tf.get_variable('biases', [out_dim], tf.float32, initializer=tf.random_uniform_initializer(-stdv, stdv))
        else:
            b = tf.get_variable('biases', [out_dim], tf.float32, initializer=tf.constant_initializer(0))

        # Append the variable to the trainable variables list
        var_list.append(w)
        var_list.append(b)

    # Do the FC operation
    output = tf.matmul(x, w) + b
    return output

def _bn(x, var_list, train_phase, scope='bn_'):
    """
    Batch normalization on convolutional maps.
    Args:

    Return:
    """
    n_out = x.get_shape().as_list()[3]
    with tf.variable_scope(scope):
        beta = tf.get_variable('beta', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable('gamma', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        var_list.append(beta)
        var_list.append(gamma)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        
        mean, var = tf.cond(train_phase,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed

def _residual_block(x, trainable_vars, train_phase, apply_relu=True, scope="unit"):
    """
    ResNet block when the number of channels across the skip connections are the same
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        shortcut = x
        x = _conv(x, 3, in_channels, 1, trainable_vars, scope='conv_1')
        x = _bn(x, trainable_vars, train_phase, scope="bn_1")
        x = tf.nn.relu(x)
        x = _conv(x, 3, in_channels, 1, trainable_vars, scope='conv_2')
        x = _bn(x, trainable_vars, train_phase, scope="bn_2")

        x = x + shortcut
        if apply_relu == True:
            x = tf.nn.relu(x)

    return x


def _residual_block_first(x, out_channels, strides, trainable_vars, train_phase, apply_relu=True, scope="unit", is_ATT_DATASET=False):
    """
    A generic ResNet Block
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        # Figure out the shortcut connection first
        if in_channels == out_channels:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = _conv(x, 1, out_channels, strides, trainable_vars, scope="shortcut")
            if not is_ATT_DATASET:
                shortcut = _bn(shortcut, trainable_vars, train_phase, scope="bn_0")

        # Residual block
        x = _conv(x, 3, out_channels, strides, trainable_vars, scope="conv_1")
        x = _bn(x, trainable_vars, train_phase, scope="bn_1")
        x = tf.nn.relu(x)
        x = _conv(x, 3, out_channels, 1, trainable_vars, scope="conv_2")
        x = _bn(x, trainable_vars, train_phase, scope="bn_2")

        x = x + shortcut
        if apply_relu:
            x = tf.nn.relu(x)

    return x



def resnet18_conv_feedforward(h, kernels, filters, strides,out_dim,train_phase,is_ATT_DATASET=False,net_type='RESNET-S'):
        """
        Forward pass through a ResNet-18 network

        Returns:
            Logits of a resnet-18 conv network
        """
        trainable_vars, H = [], []

        # Conv1
        h = _conv(h, kernels[0], filters[0], strides[0], trainable_vars, scope='conv_1')
        h = _bn(h, trainable_vars, train_phase, scope='bn_1')
        h = tf.nn.relu(h)
        H.append(h)
        # Conv2_x
        h = _residual_block(h, trainable_vars, train_phase, scope='conv2_1')
        H.append(h)
        h = _residual_block(h, trainable_vars, train_phase, scope='conv2_2')
        H.append(h)
        # Conv3_x
        h = _residual_block_first(h, filters[2], strides[2], trainable_vars, train_phase, scope='conv3_1', is_ATT_DATASET=is_ATT_DATASET)
        H.append(h)
        h = _residual_block(h, trainable_vars, train_phase, scope='conv3_2')
        H.append(h)
        # Conv4_x
        h = _residual_block_first(h, filters[3], strides[3], trainable_vars, train_phase, scope='conv4_1', is_ATT_DATASET=is_ATT_DATASET)
        H.append(h)
        h = _residual_block(h, trainable_vars, train_phase, scope='conv4_2')
        H.append(h)
        # Conv5_x
        h = _residual_block_first(h, filters[4], strides[4], trainable_vars, train_phase, scope='conv5_1', is_ATT_DATASET=is_ATT_DATASET)
        H.append(h)
        h = _residual_block(h, trainable_vars, train_phase, scope='conv5_2')

        # Apply average pooling
        h = tf.reduce_mean(h, [1, 2])

        H.append(h)
        print('output layer shape',h.shape)
        if net_type == 'RESNET-S':
            logits = _fc(h, out_dim, trainable_vars, scope='fc_1', is_cifar=True)
        else:
            logits = _fc(h, out_dim, trainable_vars, scope='fc_1')
        H.append(logits)

        return H, trainable_vars