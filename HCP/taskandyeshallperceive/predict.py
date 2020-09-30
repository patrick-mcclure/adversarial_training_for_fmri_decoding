# -*- coding: utf-8 -*-
import tensorflow.contrib.layers as lays
import nibabel as nib
import tensorflow as tf
import numpy as np
import time
import pandas as pd
from pathlib import Path
from taskandyeshallperceive.models.predictor import predictor, linear, cln
from taskandyeshallperceive.util import get_batch,get_loss,csv_to_batches,zscore,hessian_vector_product
from random import shuffle,seed
from scipy.io import mmread
from scipy.sparse import spdiags
import scipy
import os
from keras_radam.training import RAdamOptimizer
import tensorflow.keras.backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow import gradients

def predict(model_dir,model_type,input_csv,output_dir,temperature,target_task,n_samples,noise_level):
    lr = 1e-3
    batch_size=1
    hessian_flg = False
    
    # The CNNs used a temperature of 1.0 and the linear model used a temperature of 100.0
    # A SmoothGrad noise level of 2.0 was used.
    
    tf.reset_default_graph()
    if hessian_flg:
        brain_atlas = nib.load('/data/SharedData/TaskPrediction/Parcellations/MNI_Glasser_HCP_v1.0_re.nii.gz').get_data()
        brain_parcels = np.unique(brain_atlas).astype(int).tolist()
        roi_masks = []
        for r in range(len(brain_parcels)-1):
            roi_img = np.expand_dims(np.expand_dims(np.equal(brain_atlas,brain_parcels[r+1]),axis=0),axis=-1)
            roi_img = roi_img.astype(float)/roi_img.sum()
            roi_masks.append(roi_img)
        n_voxels = brain_mask.sum()
        print(n_voxels)
        print(brain_mask.shape)
        brain_mask_img = np.expand_dims(np.expand_dims(brain_mask,axis=0),axis=0)
        print(brain_mask_img.shape)
    
    print("Running prediction...")
    contents, batch_per_ep = csv_to_batches(input_csv, batch_size)
    
    
    x = tf.placeholder(tf.float32, (batch_size, 91, 109, 91, 1))
    
    v = tf.placeholder(tf.float32, (len(brain_parcels)-1, batch_size, 91, 109, 91, 1))
    
    y_true = tf.placeholder(tf.int64, (batch_size))
    
    y_logits = predictor(x,training=True) * (1.0/temperature) 
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_logits)
    
    acc = tf.reduce_mean(tf.dtypes.cast(tf.equal(y_true,tf.argmax(y_logits, 1)),tf.float32))
    
    mask_true = tf.one_hot(y_true,7)
    
    probs = tf.nn.softmax(y_logits)
    
    masked_loss = tf.reduce_mean( probs * mask_true,axis = -1)

    #masked_x = tf.boolean_mask(x,b)
    
    g = tf.gradients(masked_loss,x)
    
    #v_unstacked = tf.unstack(v)
    #h = hessian_vector_product(masked_loss,[x for n_p in range(len(brain_parcels)-1)],v_unstacked)

    init = tf.global_variables_initializer()
    pred_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="predictor")
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:

        sess.run(init)
        saver.restore(sess, model_dir + "/model.ckpt")
        
        batches = [contents[i:i + batch_size] for i in range(0, len(contents), batch_size) if batch_size == len(contents[i:i + batch_size])]
        roi_masks = np.array(roi_masks)
        n=1
        a_sum=0
        l_sum=0
        accs = []
        for ii, batch in enumerate(batches):  # batches loop
            
            batch_x, batch_y_true = get_batch(batch)
            
            if target_task != None:
                batch_y_true.fill(target_task)
            
            if hessian_flg:
                l_tmp, a_tmp, g_sum, h_sum = sess.run([loss, acc, g, h], feed_dict={x: batch_x, y_true: batch_y_true, v:roi_masks})
            else:
                l_tmp, a_tmp, g_sum = sess.run([loss, acc, g], feed_dict={x: batch_x, y_true: batch_y_true})
            
            a_sum += a_tmp
            accs.append(a_tmp)
            l_sum += l_tmp[0]
            
            g_sum = g_sum[0]
            
            for k in range(n_samples-1):
                batch_x_tmp = batch_x + np.random.normal(loc=0.0, scale=noise_level, size=batch_x.shape)
                if target_task == None:
                    if hessian_flg:
                        g_tmp, h_tmp = sess.run([g, h], feed_dict={x: batch_x_tmp, y_true: batch_y_true, v:roi_masks})
                        h_sum += np.array(h_tmp)
                    else:
                        g_tmp = sess.run(g, feed_dict={x: batch_x_tmp, y_true: batch_y_true, v:roi_masks})
                else:
                    g_tmp, pr = sess.run([g,probs], feed_dict={x: batch_x_tmp, y_true: batch_y_true})
                g_sum += g_tmp[0]
                
            g_sum /= float(n_samples)
            
            if hessian_flg:
                h_sum /= float(n_samples)
            
            print('loss = {:.5f} - acc = {:.5f}'.format(l_sum/n, a_sum/n))
            n += 1 
            
            orig_img = nib.load(batch[0][0])
            img_data = np.zeros(brain_atlas.shape)
            
            output_img = nib.spatialimages.SpatialImage(
                dataobj=g_sum.squeeze(), affine=orig_img.affine, header=orig_img.header, extra=orig_img.extra)
            
            print(contents[ii][0].split('/')[-1])
            
            if target_task == None:
                nib.save(output_img,output_dir + "grads_" + contents[ii][0].split('/')[-1])
                if hessian_flg:
                    p.save(output_dir + "hessian_" + contents[ii][0].split('/')[-1].replace('.nii.gz','.npy'),h_sum)
            else:
                nib.save(output_img,output_dir + "grads_" + str(target_task) + '_' + contents[ii][0].split('/')[-1])
            print(str(ii),'/', str(len(contents)))
    np.save(output_dir + 'accs.npy',accs)