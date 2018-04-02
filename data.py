#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import cv2
import numpy as np
import tensorflow as tf

import database.loaders as db


def _on_first(func):
    """Wraps function 'func' so that it is applied only to the first argument."""
    def on_all(first, *args):
        return (func(first), *args)
    return on_all


class Database:
    """Gathers files from all databases in one dataset.

    Dataset outputs pairs (image, label), where image is of size IMAGE_SIZE
    and of type tf.float32. Label is a number from 0 to 61 representing index of
    that label in LABELS.
    """
    LABELS = CLASSES = db.LABELS
    N_CLASSES = len(CLASSES)
    IMAGE_SIZE = (28, 28)
    DATASETS = {
        'Char47K': db.Char47KLoader(hand_upscale=10, images_upscale=3),
        }

    def __init__(self, datasets='all', num_parallel_calls=3):
        self.num_parallel_calls = num_parallel_calls
        self.logger = logging.getLogger('database')
        datasets = self.DATASETS.keys() if datasets == 'all' else datasets
        self.loaders = [self.DATASETS[db] for db in datasets]

    def get_train_dataset(self):
        filenames, labels = self.load_all_files('train')
        return self.prepare_dataset(filenames, labels)

    def get_test_dataset(self):
        filenames, labels = self.load_all_files('test')
        return self.prepare_dataset(filenames, labels)

    def prepare_dataset(self, filenames, labels=None):
        """Creates dataset from filenames and labels.

        The dataset elements are[*] of form (image, label) where image
        is float32 2D tesnsor of IMAGE_SIZE and label is a number 0-61.

        [*] If labels is None, creates dataset of images only (each element is a
        single Tensor, not tuple).
        """
        data = (filenames, labels) if labels else filenames
        dataset = tf.data.Dataset.from_tensor_slices(data)
        return dataset.map(_on_first(self.load_image), self.num_parallel_calls)

    def load_image(self, filepath):
        """Reads image and resizes it. Pixel values are float32 from 0 256."""
        image_bytes = tf.read_file(filepath)
        # because decode_image doesn't return shape ?!
        image = tf.image.decode_png(image_bytes, channels=3)
        image = tf.image.resize_images(image, self.IMAGE_SIZE)
        return image

    def load_all_files(self, which, squeeze=True):
        """Gathers files from all the loaders.

        which - can be 'test', 'train' or both ['test', 'train']
        squeeze - if requested only one type as 'which', returns just the
            dictionary entry for that type

        Returns dictionary (with keys 'which') of tuples of lists:
            { 'test': ([filepath, ...], [label, ...]), 'train': ... }
        """
        which = [which] if isinstance(which, str) else which
        all_data = {train_or_test: ([], []) for train_or_test in which}
        for data in (loader.load_files(which) for loader in self.loaders):
            for train_test in which:
                all_data[train_test][0].extend(data[train_test][0])
                all_data[train_test][1].extend(data[train_test][1])
        if squeeze and len(which) == 1:
            all_data = all_data[which[0]]
        return all_data

################################################################################

# def test_sizes():
#     data = load_chars74k_files()
#     sizes = {0}
#     for img, lab in data['test']:
#         img = cv2.imread(img)
#         sizes.add(img.shape)
#     sizes.remove(0)
#     for s in sorted(sizes, key=lambda x: x[0]):
#         print(s)

if __name__ == '__main__':
    database = Database()
    train_dataset = database.get_train_dataset()
    test_dataset = database.get_test_dataset()
    next_el = train_dataset.concatenate(test_dataset).make_one_shot_iterator().get_next()
    print(next_el)

    sess = tf.Session()
    N = 1000
    i = 0
    while i < N:
        try:
            image, label = sess.run(next_el)
            cv2.imshow('image', image/256)
            if cv2.waitKey(0) in [27, ord('q')]: # 'ESCAPE' or 'q'
                break
            i += 1
        except tf.errors.OutOfRangeError:
            break
