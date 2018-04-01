#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf

import cnn_model
import database.data

################################################################################

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

    def predict_input_fn(filenames):
        images = list(map(database.data.load_image, filenames))
        return tf.data.Dataset.from_tensor_slices(images).batch(args.batch_size)

    if args.train:
        estimator.train(train_input_fn)

    if args.eval:
        results = estimator.evaluate(eval_input_fn)
        print('Test data accuracy: %.3f' % results['accuracy'])

    if args.predict:
        predictions = estimator.predict(predict_input_fn)
        filenames = list(args.predict)
        print('Predictions:')
        for prediction_dict in predictions:
            for i, pred_ij in enumerate(prediction_dict['predictions']):
                # i       - i-th image in batch
                # pred_ij - j-th class (prediction) for i-th image
                filename = filenames.pop(0)
                label = database.data.CLASSES[pred_ij]
                probability = prediction_dict['probabilities'][pred_ij]
                print('%s: %s (%.2f %%)' % (filename, label, probability))
        assert len(filenames) == 0


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
