# -*- coding: utf-8 -*-
"""Utilities."""
import numpy as np
import nibabel as nib
import pandas as pd
import sys
import tensorflow as tf
from pathlib import Path
import argparse
import os
from nilearn import image, datasets
from nibabel import processing

#import nibabel.spatialimages.HeaderDataError

# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))

def check_path_is_writable(p, path_type='output csv'):
    if not os.access(p, mode=os.W_OK):
        raise ValueError(f"Do not have write access to {p}"
                         "which has been specified as the {path_type} path")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_nibload(input_path):

    try:
        nib.load(input_path).get_data()
    except Exception as e:
        print(e)
        print("Failure, could not read this file: ", input_path)
        return False
    return True


def zscore(a):
    """Return array of z-scored values."""
    a = np.asarray(a)
    std = a.std()
    if std == 0:
        std = 10**-7
    return (a - a.mean()) / std


def run_cmd(cmd):
    import subprocess
    pp = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print([v.split('//')[-1] for v in pp.stderr.decode('utf-8').splitlines()])
    return pp


def setup_exceptionhook():
    """
    Overloads default sys.excepthook with our exceptionhook handler.

    If interactive, our exceptionhook handler will invoke pdb.post_mortem;
    if not interactive, then invokes default handler.
    """
    def _pdb_excepthook(type, value, tb):
        if sys.stdin.isatty() and sys.stdout.isatty() and sys.stderr.isatty():
            import traceback
            import pdb
            traceback.print_exception(type, value, tb)
            # print()
            pdb.post_mortem(tb)
        else:
            print(
                "We cannot setup exception hook since not in interactive mode")

    sys.excepthook = _pdb_excepthook


def get_image(image_path):
    orig_img = nib.load(image_path)
    z_img = zscore(orig_img.get_data())
    return z_img, orig_img


def get_batch(batch,stats_path=None):
    # This method returns a batch and its labels.

    # input:
    #   batch: list of string address of original images.

    # output:
    #   imgs: 5D numpy array of image btaches. Specifically, it takes the shape of (batch_size, 256, 256, 256, 1)
    
    arr_in_img = []
    arr_label = []

    for x in batch:
        z_in_img,_ = get_image(x[0])
        arr_in_img.append(z_in_img)
        z_label = np.array(int(x[1]))
        arr_label.append(z_label)
    in_imgs = np.array(arr_in_img)
    in_imgs = np.expand_dims(in_imgs,axis=-1)
    labels = np.array(arr_label)
    return in_imgs, labels


def csv_to_batches(csv, batch_size):
    df = pd.read_csv(csv).dropna()
    files = list(df[df.columns[0]])
    task_ids = list(df[df.columns[2]])
    contents = []
    for i in range(len(files)):
        if os.path.exists(files[i]):
            contents += [(files[i],task_ids[i])]
    batch_per_ep = len(contents) // batch_size

    return contents, batch_per_ep

def get_loss(ae_outputs, gt_outputs, ae_inputs, ae_logstds):
    #abs loss
    recon_loss = tf.keras.backend.square(ae_outputs - gt_outputs)
    baseline_loss = tf.keras.backend.mean(tf.keras.backend.square(ae_inputs - gt_outputs))
    total_loss = tf.keras.backend.mean(recon_loss / (2.0 * (tf.square(tf.exp(ae_logstds))+1e-8)) + ae_logstds)
    #total_loss = recon_loss
    return total_loss, baseline_loss

def hessian_vector_product(ys, xs, v):
    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = gradients(ys, xs)

    # grads = xs

    assert len(grads) == length

    elemwise_products = [
        math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, [v]) if grad_elem is not None
  ]

    # Second backprop  
    grads_with_none = gradients(elemwise_products, xs)
    return_grads = [
        grad_elem if grad_elem is not None else tf.zeros_like(x) for x, grad_elem in zip(xs, grads_with_none)]
  
    return return_grads