#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import tensorflow as tf

import gui
import data
import cnn_model
import log

################################################################################

# (note) for gpu usage monitoring: optirun nvidia-smi -l 2

DEFAULTS = {
    'epochs': 1,
    'batch_size': 100,
}

parser = argparse.ArgumentParser(description='TODO')
parser.add_argument('-b', '--batch_size', type=int, default=DEFAULTS['batch_size'],
    help='Number of images in a batch')
parser.add_argument('-e', '--epochs', type=int,  default=DEFAULTS['epochs'],
    help='Number of epochs of training (one epoch means traversing the whole training database)')
parser.add_argument('-T', '--train', action='store_true',
    help='Perform training using the training database')
parser.add_argument('-E', '--eval', action='store_true',
    help='Evaluate accurancy on the test database')
parser.add_argument('-P', '--predict', nargs='+', metavar='IMG_FILE',
    help='Perform prediction on given image files')
parser.add_argument('-G', '--gui', action='store_true',
    help='Run interactive prediction GUI (discards other options)')
parser.add_argument('-v', '--verbose', action='count', default=0,
    help='Increase verbosity ERROR -> WARN -> INFO -> DEBUG (for each usage of this argument)')


def main():
    args = parser.parse_args()
    if not (args.train or args.eval or args.predict or args.gui):
        print('No action specified (see --help).')
        return

    # configure verbosity
    default = tf.logging.ERROR
    verbosity = max(tf.logging.DEBUG, default - args.verbose * tf.logging.DEBUG)
    tf.logging.set_verbosity(verbosity)
    log.setLevel(verbosity)

    database = data.Database()
    estimator = cnn_model.get_estimator()

    if args.gui:
        print('Using CPU only for better performance in GUI mode')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        gui.runApp(estimator)

    def train_input_fn():
        return database.get_train_dataset().shuffle(10000).batch(
            args.batch_size).repeat(args.epochs).prefetch(1)

    def eval_input_fn():
        return database.get_test_dataset().batch(args.batch_size).prefetch(1)

    def predict_input_fn():
        return database.prepare_dataset(args.predict).batch(args.batch_size)

    if args.train:
        estimator.train(train_input_fn)

    if args.eval:
        results = estimator.evaluate(eval_input_fn)
        print('Test data accuracy: %.3f' % results['accuracy'])

    if args.predict:
        predictions = estimator.predict(predict_input_fn)
        common_path = os.path.split(os.path.commonprefix(args.predict))[0]
        filenames = [os.path.relpath(path, start=common_path) for path in args.predict]
        max_filename_len = max(len(name) for name in filenames)

        print('Predictions:')
        for filename, prediction_dict in zip(filenames, predictions):
            pi = prediction_dict['predictions']
            label = database.CLASSES[pi]
            probability = prediction_dict['probabilities'][pi]
            print('{name:>{nlen}}: {lab} ({prob:6.2f} %)'.format(name=filename,
                nlen=max_filename_len, lab=label, prob=probability * 100))


if __name__ == '__main__':
    main()
