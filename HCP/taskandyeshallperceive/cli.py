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
        '-e', '--n-epochs', type=int, default=5,
        help="Number of training epochs")

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
    ppp.add_argument('--noise-level',required= False, help="SmoothGrad noise level.",default=0.0)
    
    
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
    _train(
        model_dir=params['model_dir'],
        input_csv=params['input_csv'],
        batch_size=params['batch_size'],
        n_epochs=params['n_epochs'],
        )

def predict(params):
    _predict(
        model_dir=params['model_dir'],
        input_csv=params['input_csv'],
        output_dir=params['output_dir'],
        temperature=params['temperature']
        target=params['target'],
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