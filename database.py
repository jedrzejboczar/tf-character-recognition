#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import string
import logging
import subprocess

import cv2
import numpy as np
import tensorflow as tf


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
    LABELS = '0123456789' + string.ascii_uppercase + string.ascii_lowercase
    CLASSES = LABELS
    N_CLASSES = len(CLASSES)
    IMAGE_SIZE = (28, 28)

    def __init__(self):
        this_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
        self.database_dir = os.path.join(this_dir, 'database')
        self.logger = logging.getLogger('database')
        self.loaders = [self.load_chars74k_files, ]
        self.num_parallel_calls = 3

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
        for data in (loader(which) for loader in self.loaders):
            for train_test in which:
                all_data[train_test][0].extend(data[train_test][0])
                all_data[train_test][1].extend(data[train_test][1])
        if squeeze and len(which) == 1:
            all_data = all_data[which[0]]
        return all_data

    def load_chars74k_files(self, which_types):
        """Loads data from Char47K database.

        It contains 78905 PNG images (font:hand:igood:ibad -> 62992:3410:4798:7705).
        The database consists of 3 groups:          (height x width x channels)
            font - images generated from fonts      (128x128x3)
            hand - images drawn on tablet           (900x1200x3)
            img_good - photos of different sizes    (from 6x16x3 to 391x539x3)
            img_bad - photos with worse quality     (from 11x7x3 to 464x325x3)

        which_types - must be a list of types ['test', 'train']

        Returns dictionary (with keys 'which_types') of tuples of lists:
            { 'test': ([filepath, ...], [label, ...]), 'train': ... }
        """
        chars74k_dir = os.path.join(self.database_dir, 'chars74k')
        result = subprocess.run(['./prepare_database.py', '--check-only'], cwd=chars74k_dir)
        if result.returncode != 0:
            logger.warn('Chars74K may be incomplete. Please check in %s.' % chars74k_dir)
        # gather data as tuple of lists ([filepaths], [labels])
        data = {train_or_test: ([], []) for train_or_test in which_types}
        for dirname in ['font/', 'hand/', 'img_bad', 'img_good']:
            for train_or_test in which_types:
                for label in self.LABELS:
                    classdir_path = os.path.join(chars74k_dir, dirname, train_or_test, label)
                    for filename in os.listdir(classdir_path):
                        filepath = os.path.join(classdir_path, filename)
                        numeric_label = self.LABELS.find(label)
                        data[train_or_test][0].append(filepath)
                        data[train_or_test][1].append(numeric_label)
        return data

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
