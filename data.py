#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import tensorflow as tf

import log
import database.loaders


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
    LABELS = CLASSES = database.loaders.LABELS
    N_CLASSES = len(CLASSES)
    IMAGE_SIZE = (28, 28)
    DATASETS = {
        'Char47K': database.loaders.Char47K(hand_upscale=10, images_upscale=3),
        }

    def __init__(self, num_parallel_calls=3):
        self.n_parallel = num_parallel_calls
        self.logger = log.getLogger('database')
        self.loaders = list(self.DATASETS.values())

    def get_train_dataset(self):
        return self.get_dataset(is_training=True)

    def get_test_dataset(self):
        return self.get_dataset(is_training=False)

    def get_dataset(self, is_training):
        if is_training:
            datasets = [loader.get_train_dataset() for loader in self.loaders]
        else:
            datasets = [loader.get_test_dataset() for loader in self.loaders]
        dataset = datasets[0]
        for ds in datasets[1:]:
            dataset = dataset.concatenate(ds)
        # shuffle much once at the begging and cache results (training only)
        if is_training:  # 100k should take at most 10MB in RAM
            dataset = dataset.shuffle(100000, reshuffle_each_iteration=False).cache()
        # load images
        dataset = dataset.map(_on_first(self.load_image), num_parallel_calls=self.n_parallel)
        # for now leave rest of operations for others, e.g.:
        #   .cache().shuffle(small_buffer).batch(size).repeat(epochs).prefetch(1)
        #   or without .cache as it will take much RAM because caching whole images
        return dataset

    def from_files(self, filepaths, labels=None):
        """Creates dataset from filenames and labels (e.g. for predict_input_fn())

        The dataset elements are[*] of form (image, label) where image
        is float32 2D tesnsor of IMAGE_SIZE and label is a number 0-61.

        [*] If labels is None, creates dataset of images only (each element is a
        single Tensor, not tuple).
        """
        data = (filepaths, labels) if labels else filepaths
        dataset = tf.data.Dataset.from_tensor_slices(data)
        return dataset.map(_on_first(self.load_image), num_parallel_calls=self.n_parallel)

    def load_image(self, filepath):
        """Reads image and resizes it. Pixel values are float32 from 0 256."""
        image_bytes = tf.read_file(filepath)
        # because decode_image doesn't return shape ?!
        image = tf.image.decode_png(image_bytes, channels=3)
        image = tf.image.resize_images(image, self.IMAGE_SIZE)
        return image

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
