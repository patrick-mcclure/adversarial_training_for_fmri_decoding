import tensorflow as tf
import scipy
import numpy as np
import math
import time
from .bayesian_dropout import bernoulli_dropout
from scipy.sparse import find

from taskandyeshallperceive.models.wn_conv import conv3d as conv

relu = tf.nn.relu

def pool(input_, name="pool3d"):
    return tf.nn.avg_pool3d(input_,[1,2,2,2,1],[1,1,1,1,1],'SAME',name=name)

def conv3d(input_, output_dim, ks=3, s=1, d=1, padding='SAME', name="conv3d"):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return conv(input_, output_dim, ks, s, padding=padding, kernel_initializer=tf.keras.initializers.he_normal(),dilation_rate=d)
    
def predictor(inputs,training):
    with tf.variable_scope('predictor',reuse=tf.AUTO_REUSE):
        # image is 256 x 256 x input_c_dim
        
        h1 = pool(relu(conv3d(inputs,32,name='c3d1')))
        print(h1.get_shape().as_list())
        h2 = pool(relu(conv3d(h1,32,name='c3d2')))
        print(h2.get_shape().as_list())
        h3 = pool(relu(conv3d(h2,64,name='c3d3')))
        print(h3.get_shape().as_list())
        h4 = pool(relu(conv3d(h3,64,name='c3d4')))
        print(h4.get_shape().as_list())
        h5 = pool(relu(conv3d(h3,64,name='c3d5')))
        print(h5.get_shape().as_list())
        h5_f = tf.contrib.layers.flatten(h5)
        print(h5_f.get_shape().as_list())
        y = tf.contrib.layers.fully_connected(h5_f,7,activation_fn=None,reuse=tf.AUTO_REUSE,scope='f1d1')
        print(y.get_shape().as_list())
        return y
    
def cln(inputs,training):
    with tf.variable_scope('cln',reuse=tf.AUTO_REUSE):
        # image is 256 x 256 x input_c_dim
        c = 1e-7
        inputs_f = tf.contrib.layers.flatten(inputs)
        h_f = tf.contrib.layers.fully_connected(inputs_f,75,activation_fn=None,weights_regularizer=tf.keras.regularizers.l2(l=c),reuse=tf.AUTO_REUSE,scope='fc1')
        y = tf.contrib.layers.fully_connected(h_f,7,activation_fn=None,weights_regularizer=tf.keras.regularizers.l2(l=c),reuse=tf.AUTO_REUSE,scope='fc2')
        print(y.get_shape().as_list())
        return y
    
def linear(inputs,training):
    with tf.variable_scope('linear',reuse=tf.AUTO_REUSE):
        # image is 256 x 256 x input_c_dim
        inputs_f = tf.contrib.layers.flatten(inputs)
        y = tf.contrib.layers.fully_connected(inputs_f,7,weights_regularizer=tf.keras.regularizers.l2(l=1e-7),activation_fn=None,reuse=tf.AUTO_REUSE,scope='l1')
        return y