# -*- coding: utf-8 -*-
import tensorflow.contrib.layers as lays
import nibabel as nib
import tensorflow as tf
import numpy as np
import time
import pandas as pd
from random import uniform
from pathlib import Path
from taskandyeshallperceive.models.predictor import predictor, linear, cln
from taskandyeshallperceive.util import get_batch,csv_to_batches,zscore
from random import shuffle,seed
from scipy.io import mmread
from scipy.sparse import spdiags
from scipy.sparse import find
import scipy
import os
from keras_radam.training import RAdamOptimizer
# from delorean.util import zscore

model_type = 'nn'
n_gpus = 8
epsilon = 0.95 #9.5 #95
l2_coeff = 1e-9
sphere = True

def get_noise(shape,epsilon):
    n_noise_samples = shape[0]
    shape.pop(0)
    noise = []
    for n in range(n_noise_samples):
        noise_tmp = np.random.normal(size=shape)
        u = np.random.uniform(size=())
        d = float(np.prod(np.array(shape)))
        noise_tmp = epsilon * u**(1.0/d) * noise_tmp / (np.linalg.norm(noise_tmp)+1e-16)
        noise.append(noise_tmp)
    return np.array(noise)

def make_parallel(fn, num_gpus,batch_size,epsilon, **kwargs):
    
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                fn(**{k : v[i] for k, v in in_splits.items()})

def model(x,y_true,mask_true):
    
    if model_type == 'nn':
        y_logits = predictor(x,training=True)
    elif model_type == 'linear':
        y_logits = linear(x,training=True)
    elif model_type == 'cln':
        y_logits = cln(x,training=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_logits)
    if model_type == 'nn':
        kernels = tf.get_collection('kernels')
        ss_l2 = 0.0
        for kernel in kernels:
            ss_l2 += tf.reduce_sum(tf.square(kernel))
        loss += l2_coeff * tf.sqrt(ss_l2)
    acc = tf.reduce_mean(tf.dtypes.cast(tf.equal(y_true,tf.argmax(y_logits, 1)),tf.float32))
    masked_loss = tf.reduce_mean(tf.nn.softmax(y_logits) * mask_true,axis = -1)
    grad = tf.gradients(masked_loss,x)
    
    tf.add_to_collection('total_loss', loss)
    tf.add_to_collection('total_acc', acc)
    tf.add_to_collection('total_grad', grad)

def train(model_dir,input_csv,batch_size,n_epochs):
    lr = 1e-3
    contents, batch_per_ep = csv_to_batches(input_csv, batch_size)
    
    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        
        x = tf.placeholder(tf.float32, (batch_size, 91, 109, 91, 1))
        
        y_true = tf.placeholder(tf.int64, (batch_size))
        
        mask_true = tf.one_hot(y_true,7)
        
    make_parallel(model, n_gpus, batch_size, epsilon, x=x, y_true=y_true,mask_true=mask_true)
    
    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        
        total_loss_collection = tf.add_n(tf.get_collection('total_loss')) / float(n_gpus)
        total_acc_collection = tf.add_n(tf.get_collection('total_acc')) / float(n_gpus)
        total_grad_collection = tf.get_collection('total_grad')
        
        train_gen = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss_collection)
        #RAdamOptimizer(learning_rate=lr).minimize(total_loss_collection)

    # initialize the network
    init = tf.global_variables_initializer()
    
    if model_type == 'nn':
        pred_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="predictor")
    elif model_type == 'linear':
        pred_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="linear")
    elif model_type == 'cln':
        pred_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="cln")
    print(pred_vars)
    saver = tf.train.Saver(pred_vars)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    
    with tf.Session(config=config) as sess:

        sess.run(init)
        n_m = 0
        t = 0
        if n_m ==0:
            n_ep = n_epochs
        else:
            n_ep = round(float(n_epochs)/float(n_m))
        for ep in range(n_ep):  # epochs loop
            time2 = time.time()
            seed(1)
            shuffle(contents)
            batches = [contents[i:i + batch_size] for i in range(0, len(contents), batch_size) if batch_size == len(contents[i:i + batch_size])]
            i = 0
            for batch in batches:  # batches loop
                if not i%3:
                        save_path = saver.save(sess, model_dir + "/model.ckpt", write_meta_graph=False)
                time1 = time.time()
                batch_x, batch_y_true = get_batch(batch)
                noise = np.zeros((batch_size, 91, 109, 91, 1))
                if sphere:
                    noise_sphere = get_noise([batch_size, 91, 109, 91, 1],epsilon)
                if n_m == 0:
                    if sphere:
                        batch_x += noise_sphere
                    _, l, a = sess.run([train_gen, total_loss_collection, total_acc_collection], feed_dict={x: batch_x, y_true: batch_y_true})
                    print('Epoch: {} - loss = {:.5f} - accuracy = {:.5f}'.format((ep + 1), l[0], a))
                
                else:
                    for m in range(n_m):
                        if sphere:
                            batch_x += noise_sphere
                        _, l, a = sess.run([train_gen, total_loss_collection, total_acc_collection], feed_dict={x: batch_x+noise, y_true: batch_y_true})
                        g = sess.run(total_grad_collection, feed_dict={x: batch_x+noise, y_true: batch_y_true})
                        g = np.array(g)
                        g = np.reshape(g,(batch_size, 91, 109, 91, 1))
                        
                        for k in range(g.shape[0]):
                            random_epsilon = np.random.uniform(0,epsilon,())
                            if np.linalg.norm(g[k]) != 0:
                                noise[k] -= epsilon * g[k] / np.linalg.norm(g[k])
                            noise_norm = np.linalg.norm(noise[k])
                            if noise_norm > float(epsilon):
                                noise[k] = epsilon * noise[k] / noise_norm
                    
                        print('Epoch: {} - loss = {:.5f} - accuracy = {:.5f}'.format((ep + 1), l[0], a))
                t += 1
                i = i + batch_size
                print('1 batch took ' + str(time.time() - time1) + ' seconds')
            print('1 epoch took ' + str(time.time() - time2) + ' seconds')
        #file.close()
        # save model 
        save_path = saver.save(sess, model_dir + "/" + model_type + ".ckpt", write_meta_graph=False)

