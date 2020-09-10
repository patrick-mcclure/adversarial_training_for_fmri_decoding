# -*- coding: utf-8 -*-
import tensorflow.contrib.layers as lays
import nibabel as nib
import tensorflow as tf
import numpy as np
import time
import pandas as pd
from random import uniform
from pathlib import Path
from taskandyeshallperceive.models.predictor import predictor
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
def make_parallel(fn, num_gpus,batch_size,epsilon, **kwargs):
    
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                fn(i,batch_size,epsilon,**{k : v[i] for k, v in in_splits.items()})

def model(i,batch_size,epsilon,x,y_true):
        
    x_v = tf.Variable(np.zeros((int(batch_size/n_gpus), 91, 109, 91, 1)),trainable=False,dtype=tf.float32, name='x_v_' + str(i))
        
    y_true_v = tf.Variable(np.zeros((int(batch_size/n_gpus))),trainable=False,dtype=tf.int64, name='y_true_v_' + str(i))
        
    x_set = x_v.assign(x)
    y_true_set = y_true_v.assign(y_true)
    
    mask_true = tf.one_hot(y_true_v,7)
        
    noise = tf.Variable(np.zeros((int(batch_size/n_gpus), 91, 109, 91, 1)),trainable=False,dtype=tf.float32,name='noise_' + str(i))
    x_noisy = x_v+noise
    y_logits = predictor(x_noisy,training=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_v, logits=y_logits)
    acc = tf.reduce_mean(tf.dtypes.cast(tf.equal(y_true_v,tf.argmax(y_logits, 1)),tf.float32))
    masked_loss = tf.reduce_mean(tf.nn.softmax(y_logits) * mask_true,axis = -1)
    grad = tf.gradients(masked_loss,x_v)
    random_epsilon = tf.random.uniform((),minval=0,maxval=epsilon,dtype=tf.float32)
    normalized_grad = random_epsilon * grad[0] / tf.norm(grad)
    updated_noise = noise - normalized_grad
    noise_norm = tf.norm(updated_noise)
    normalized_noise = tf.cond(noise_norm > epsilon,lambda:epsilon*updated_noise/noise_norm,lambda:updated_noise)
    noise_update = noise.assign(normalized_noise)
    noise_reset = noise.assign(np.zeros((int(batch_size/n_gpus), 91, 109, 91, 1)))
    tf.add_to_collection('total_loss', loss)
    tf.add_to_collection('total_acc', acc)
    tf.add_to_collection('total_grad', grad)
    tf.add_to_collection('noise_resets', noise_reset)
    tf.add_to_collection('noise_updates', noise_reset)
    tf.add_to_collection('x_sets', x_set)
    tf.add_to_collection('y_sets', y_true_set)
def train(model_dir,input_csv,batch_size,n_epochs):
    epsilon = 475
    lr = 1e-3
    contents, batch_per_ep = csv_to_batches(input_csv, batch_size)
    
    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        
        x = tf.placeholder(tf.float32, (batch_size, 91, 109, 91, 1))
        
        y_true = tf.placeholder(tf.int64, (batch_size))
        
    make_parallel(model, n_gpus, batch_size, epsilon, x=x, y_true=y_true)
    
    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        
        total_loss_collection = tf.add_n(tf.get_collection('total_loss')) / float(n_gpus)
        total_acc_collection = tf.add_n(tf.get_collection('total_acc')) / float(n_gpus)
        total_grad_collection = tf.get_collection('total_grad')
        noise_resets = tf.get_collection('noise_resets')
        noise_updates = tf.get_collection('noise_updates')
        x_sets = tf.get_collection('x_sets')
        y_sets = tf.get_collection('y_sets')
        train_gen = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss_collection)
        #RAdamOptimizer(learning_rate=lr).minimize(total_loss_collection)

    # initialize the network
    init = tf.global_variables_initializer()
    
    pred_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="predictor")
    print(pred_vars)
    saver = tf.train.Saver(pred_vars)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    
    with tf.Session(config=config) as sess:

        sess.run(init)
        n_m = 4
        t = 0
        for ep in range(round(float(n_epochs)/float(n_m))):  # epochs loop
            time2 = time.time()
            seed(1)
            shuffle(contents)
            batches = [contents[i:i + batch_size] for i in range(0, len(contents), batch_size) if batch_size == len(contents[i:i + batch_size])]
            i = 0
            for batch in batches:  # batches loop
                sess.run(noise_resets)
                if not i%3:
                        save_path = saver.save(sess, model_dir + "/model.ckpt", write_meta_graph=False)
                time1 = time.time()
                batch_x, batch_y_true = get_batch(batch)
                sess.run([x_sets,y_sets], feed_dict={x: batch_x, y_true: batch_y_true})
                for m in range(n_m):    
                    _, l, a = sess.run([train_gen, total_loss_collection, total_acc_collection])
                    
                    sess.run(noise_updates)
                    
                    print('Epoch: {} - loss = {:.5f} - accuracy = {:.5f}'.format((ep + 1), l[0], a))
                t += 1
                i = i + batch_size
                print('1 batch took ' + str(time.time() - time1) + ' seconds')
            print('1 epoch took ' + str(time.time() - time2) + ' seconds')
        #file.close()
        # save model 
        save_path = saver.save(sess, model_dir + "/" + model_type + ".ckpt", write_meta_graph=False)

