#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import tensorflow as tf

import cnn_model
import database.data

################################################################################

# (note) for gpu usage monitoring: optirun nvidia-smi -l 2

DEFAULTS = {
    'epochs': 1,
    'batch_size': 100,
}

parser = argparse.ArgumentParser(description='TODO')
parser.add_argument('-b', '--batch_size', type=int, default=DEFAULTS['batch_size'])
parser.add_argument('-e', '--epochs', type=int,  default=DEFAULTS['epochs'])
parser.add_argument('-T', '--train', action='store_true')
parser.add_argument('-E', '--eval', action='store_true')
parser.add_argument('-P', '--predict', nargs='+', metavar='image_file')


def main():
    args = parser.parse_args()
    if not (args.train or args.eval or args.predict):
        print('No action specified (see --help).')
        return

    # train_dataset, test_dataset = database.data.load_datasets()
    estimator = cnn_model.get_estimator()

    def train_input_fn():
        train_dataset, test_dataset = database.data.load_datasets()
        return train_dataset.cache().shuffle(50000).batch(
            args.batch_size).repeat(args.epochs).prefetch(1)

    def eval_input_fn():
        train_dataset, test_dataset = database.data.load_datasets()
        return test_dataset.batch(args.batch_size).prefetch(1)

    def predict_input_fn():
        filenames = args.predict
        images = list(map(database.data.load_image, filenames))
        return tf.data.Dataset.from_tensor_slices(images).batch(args.batch_size)

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
            label = database.data.CLASSES[pi]
            probability = prediction_dict['probabilities'][pi]
            print('{name:>{nlen}}: {lab} ({prob:6.2f} %)'.format(name=filename,
                nlen=max_filename_len, lab=label, prob=probability * 100))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
