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
    IMAGE_SIZE = (94, 94)
    DATASETS = {
        'Char47K': database.loaders.Char47K(dirs=['font', 'hand', 'img_good'],
            hand_upscale=20, images_upscale=8),
        }

    def __init__(self, distortions=True, num_parallel_calls=3):
        self.n_parallel = num_parallel_calls
        self.distortions = distortions
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
        # add distortions
        if self.distortions:
            dataset = dataset.apply(self.add_distortions)
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

    def add_distortions(self, dataset):
        """Adds distortions to the dataset (each element should be (image, label))"""
        distortions = [self.negative]
        self.logger.info('Applying %d distortions (this will increase the dataset that many times)')
        def flat_map_func(image, label):
            new_images = [image] + [distort(image) for distort in distortions]
            return tf.data.Dataset.from_tensor_slices((new_images, tf.tile([label], [len(new_images)])))
        dataset = dataset.interleave(flat_map_func, cycle_length=1)
        return dataset

    def load_image(self, filepath):
        """Reads image and resizes it. Pixel values are float32 from 0 256."""
        image_bytes = tf.read_file(filepath)
        # because decode_image doesn't return shape ?!
        image = tf.image.decode_png(image_bytes, channels=1)
        image = tf.image.resize_images(image, self.IMAGE_SIZE)
        return image

    def negative(self, image):
        return tf.constant(255.0) - image

    def rand_rotate(self, image):
        # rotate only by 90 or 270 (no upside-down images); n in {1, 3}
        # [0, 1] * 2 = [0, 2] + 1 = [1, 3]
        n_times_90 = tf.random_uniform([], minval=0, maxval=1+1, dtype=tf.int32) * 2 + 1
        return  tf.image.rot90(image, n_times_90)


################################################################################

def test_sizes():
    from PIL import Image

    db = database.loaders.Char47K()
    datasets = {}
    for dir in ['font', 'hand', 'img_bad', 'img_good']:
        db = database.loaders.Char47K(dirs=[dir])
        datasets[dir] = db.get_test_dataset().concatenate(db.get_train_dataset())

    def get_sizes(ds):
        next = ds.prefetch(10).make_one_shot_iterator().get_next()
        sizes = []
        i = 0
        with tf.Session() as sess:
            while True:
                try:
                    imgfile, lab = sess.run(next)
                    with Image.open(imgfile) as img:
                        sizes.append(img.size)
                    i += 1
                    if i  % 100 == 0:
                        print(i, end='\r', flush=True)
                except tf.errors.OutOfRangeError:
                    break
        print()
        return sizes

    params = {}
    for dir, sizes in ((dir, get_sizes(ds)) for (dir, ds) in datasets.items()):
        sizes = np.array(sizes)
        params[dir] = {}
        params[dir]['sizes'] = sizes.copy()
        params[dir]['mean_x'] = np.mean(sizes[:, 0])
        params[dir]['mean_y'] = np.mean(sizes[:, 1])
        params[dir]['var_x'] = np.var(sizes[:, 0])
        params[dir]['var_y'] = np.var(sizes[:, 1])
        sizes = sizes[:, 0] * sizes[:, 1]
        params[dir]['mean_size'] = np.mean(sizes)
        params[dir]['var_size'] = np.var(sizes)

    for d in params.keys():
        print('### %s:' % d.upper())
        for key in params[d].keys():
            if key[:4] in ['mean', 'var_']:
                print(' %10s: %.3f' % (key, params[d][key]))
    return params


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
            print('label', database.LABELS[label], end='\r')
            cv2.imshow('image', image/256)
            if cv2.waitKey(0) in [27, ord('q')]: # 'ESCAPE' or 'q'
                break
            i += 1
        except tf.errors.OutOfRangeError:
            break
