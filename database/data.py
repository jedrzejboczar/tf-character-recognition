#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import string
import logging
import subprocess

import cv2
import numpy as np
import tensorflow as tf

################################################################################

# Labels in database (in processing they are index numbers of these)
CLASSES = LABELS = '0123456789' + string.ascii_uppercase + string.ascii_lowercase
N_CLASSES = len(CLASSES)
IMAGE_SIZE = (28, 28)

logger = logging.getLogger('database')
database_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

################################################################################

def load_datasets(add_distortions=False, n_parallel=3):
    """
    Returns (train_dataset, test_dataset) with elements (image, label) where
    image is float32 2D tesnsor of IMAGE_SIZE and label is a number 0-61.
    Training dataset is shuffled and reapeated over and over.
    """
    # get data as (filepath_list, label_list)
    all_data = {'train': ([], []), 'test': ([], [])}
    loaders = [load_chars74k_files]
    for data in (loader() for loader in loaders):
        for train_test in ['train', 'test']:
            # convert from [(filepath, label), ...] to ([filepath, ...], [label, ...])
            filenames, labels = zip(*data[train_test])
            all_data[train_test][0].extend(filenames)
            all_data[train_test][1].extend(labels)
    # create the datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(all_data['train'])
    test_dataset = tf.data.Dataset.from_tensor_slices(all_data['test'])
    # prepare datasets
        # .apply(tf.contrib.data.shuffle_and_repeat(80000)) \
    train_dataset = train_dataset \
        .map(_on_first(load_image), num_parallel_calls=n_parallel)
    test_dataset = test_dataset \
        .map(_on_first(load_image), num_parallel_calls=n_parallel)
    # do not batch the datasets
    return train_dataset, test_dataset

def load_image(filepath):
    """Reads image and resizes it. Pixel values are float32 from 0 256."""
    image_bytes = tf.read_file(filepath)
    # because decode_image doesn't return shape ?!
    image = tf.image.decode_png(image_bytes, channels=3)
    image = tf.image.resize_images(image, IMAGE_SIZE)
    return image

def load_chars74k_files():
    """
    Loads data from Char47K database.
    It contains 78905 PNG images (font:hand:igood:ibad -> 62992:3410:4798:7705).
    The database consists of 3 groups:          (height x width x channels)
        font - images generated from fonts      (128x128x3)
        hand - images drawn on tablet           (900x1200x3)
        img_good - photos of different sizes    (from 6x16x3 to 391x539x3)
        img_bad - photos with worse quality     (from 11x7x3 to 464x325x3)

    """
    chars74k_dir = os.path.join(database_dir, 'chars74k')
    result = subprocess.run(['./prepare_database.py', '--check-only'], cwd=chars74k_dir)
    if result.returncode != 0:
        logger.warn('Chars74K may not be complete. Please check in %s.' % chars74k_dir)
    # gather data as lists of tuples (filepath, label)
    data = {'train': [], 'test': []}
    for dirname in ['font/', 'hand/', 'img_bad', 'img_good']:
        for train_or_test in ['test', 'train']:
            for label in LABELS:
                classdir_path = os.path.join(chars74k_dir, dirname, train_or_test, label)
                for filename in os.listdir(classdir_path):
                    filepath = os.path.join(classdir_path, filename)
                    numeric_label = LABELS.find(label)
                    data[train_or_test].append((filepath, numeric_label))
    return data

def _on_first(func):
    """Wrap function 'func' so that it is applied only to first argument."""
    def on_all(first, *args):
        return (func(first), *args)
    return on_all


################################################################################

def test_sizes():
    data = load_chars74k_files()
    sizes = {0}
    for img, lab in data['test']:
        img = cv2.imread(img)
        sizes.add(img.shape)
    sizes.remove(0)
    for s in sorted(sizes, key=lambda x: x[0]):
        print(s)

if __name__ == '__main__':
    train_dataset, test_dataset = load_datasets()
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
