#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import string
import logging
import subprocess

import cv2
import numpy as np
import tensorflow as tf


# Labels in database (in processing they are index numbers of these)
LABELS = '0123456789' + string.ascii_uppercase + string.ascii_lowercase
IMAGE_SIZE = (100, 100)

logger = logging.getLogger('database')
database_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))


def load_datasets(resize_to=None, add_distortions=False, n_parallel=3):
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
    train_dataset = train_dataset.map(load_image_record, num_parallel_calls=n_parallel)
    test_dataset = test_dataset.map(load_image_record, num_parallel_calls=n_parallel)
    return train_dataset, test_dataset

def load_image_record(filepath, label, resize_to=None):
    """Reads image and resizes it. Pixel values are float32 from 0 256."""
    image_bytes = tf.read_file(filepath)
    image = tf.image.decode_png(image_bytes)  # because decode_image doesn't return shape ?!
    image = tf.image.resize_images(image, IMAGE_SIZE)
    return image, label

def load_chars74k_files():
    """
    Loads data from Char47K database, which consists of 3 groups:
                                                (height x width x channels)
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
    next_el = test_dataset.make_one_shot_iterator().get_next()
    print(next_el)

    sess = tf.Session()
    N = 100
    i = 0
    while i < N:
        try:
            image, label = sess.run(next_el)
            # print(image)
            cv2.imshow('image', image/256)
            cv2.waitKey(0)
            i += 1
        except tf.errors.OutOfRangeError:
            break
