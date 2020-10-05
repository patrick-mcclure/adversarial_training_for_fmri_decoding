# -*- coding: utf-8 -*-
"""Main command-line interface to taskandyeshallperceive."""

import argparse
import sys

from taskandyeshallperceive.train import train as _train
from taskandyeshallperceive.predict import predict as _predict

from taskandyeshallperceive.util import str2bool
from taskandyeshallperceive.util import setup_exceptionhook
import taskandyeshallperceive
from pathlib import Path

def create_parser():
    """Return argument parser for taskandyeshallperceive training interface."""
    p = argparse.ArgumentParser()

    p.add_argument('--debug', action='store_true', dest='debug',
                    help='Do not catch exceptions and show exception '
                    'traceback')


    subparsers = p.add_subparsers(
        dest="subparser_name", title="subcommands",
        description="valid subcommands")

    # Training subparser
    tp = subparsers.add_parser('train', help="Train models")

    m = tp.add_argument_group('model arguments')
    m.add_argument(
        '--model-dir', required=True,
        help="Directory in which to save model checkpoints. If an existing"
             " directory, will resume training from last checkpoint. If not"
             " specified, will use a temporary directory.")

    t = tp.add_argument_group('train arguments')
    t.add_argument(
        '--input-csv', required=True,
        help="Path to CSV of features, labels for training.")
    t.add_argument(
        '-o', '--optimizer', required=False,
        help="Optimizer to use for training")
    t.add_argument(
        '-l', '--learning-rate', required=False, type=float,default=0.01,
        help="Learning rate to use with optimizer for training")
    t.add_argument(
        '-b', '--batch-size', required=False, type=int,default=6,
        help="Number of samples per batch. If `--multi-gpu` is specified,"
             " batch is split across available GPUs.")
    t.add_argument(
        '-e', '--n-epochs', required=True, type=int,
        help="Number of training epochs")
    t.add_argument('--n-m', required=False, type=int, default=1,
        help="Number m-step minibatch updates")
    t.add_argument('--n-classes', required=True, type=int,
        help="Number of classes")
    t.add_argument('--model-type', required=True, type=str,
        help='Type of model: "cnn" or "linear"')
    t.add_argument('--epsilon', required=False, type=float,
        help="Adversarial training noise epsilon",default=0.95)
    t.add_argument('--l2-coeff', required=False, type=float,
        help="Adversarial training noise epsilon",default=1e-9)
    t.add_argument('--n-gpus', required=True, type=int,
        help="Adversarial training noise epsilon")
    t.add_argument('--radius', required=False, type=float,
        help="Radius of spherical random noise. The default is 0.0, which disables the noise.",default=0.0)    
    
    # Prediction subparser
    pp = subparsers.add_parser('predict', help="Predict using SavedModel")
    pp.add_argument('--input-csv',required=True, help="Filepath to csv containing scan paths.")
    ppp = pp.add_argument_group('prediction arguments')
    ppp.add_argument(
        '-m', '--model-dir', help="Path to directory containing the model.")
    ###
    ppp.add_argument('--output-dir',required= False, help="Name of output directory.",default=None)
    ppp.add_argument('--temperature',required= False, help="Softmax temperature.",default=1.0)
    ppp.add_argument('--target',required= False, help="Name of output directory.",default=None)
    ppp.add_argument('--n-samples',required= False, help="Number of SmoothGrad samples. A value of 1 does not apply SmoothGrad.",default=1)
    ppp.add_argument('--noise-level', type=float, help="SmoothGrad noise level.",default=0.0)
    ppp.add_argument('--n-classes', type=int,
        help="Number classes")
    ppp.add_argument('--model-type', required=True, type=str,
        help='Type of model: "cnn" or "linear"')

    
    
    return p

def parse_args(args):
    """Return namespace of arguments."""
    parser = create_parser()
    namespace = parser.parse_args(args)
    if namespace.subparser_name is None:
        parser.print_usage()
        parser.exit(1)
    return namespace

def train(params):
    print(params['model_type'])
    _train(
        model_dir=params['model_dir'],
        model_type=params['model_type'],
        input_csv=params['input_csv'],
        n_classes=params['n_classes'],
        n_m=params['n_m'],
        batch_size=params['batch_size'],
        n_epochs=params['n_epochs'],
        epsilon=params['epsilon'],
        l2_coeff=params['l2_coeff'],
        n_gpus=params['n_gpus'],
        radius=params['radius']
        )
    
def predict(params):
    _predict(
        model_dir=params['model_dir'],
        model_type=params['model_type'],
        input_csv=params['input_csv'],
        n_classes=params['n_classes'],
        output_dir=params['output_dir'],
        temperature=params['temperature'],
        target_task=params['target'],
        n_samples=params['n_samples'],
        noise_level=params['noise_level']
        )

def main(args=None):
    if args is None:
        namespace = parse_args(sys.argv[1:])
    else:
        namespace = parse_args(args)
    params = vars(namespace)

    if params['debug']:
        setup_exceptionhook()


    if params['subparser_name'] == 'train':
        train(params=params)

    if params['subparser_name'] == 'predict':
        predict(params=params)

if __name__ == '__main__':
    main()