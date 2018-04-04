#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import string
import logging
import subprocess
import numpy as np
import tensorflow as tf

LABELS = '0123456789' + string.ascii_uppercase + string.ascii_lowercase


class AbstractDatasetLoader:
    """
    Class that creates a dataset of tuples (filepath, label).
    """
    def __init__(self):
        this_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
        self.database_dir = os.path.join(this_dir)
        self.logger = logging.getLogger('dataset_loaders')

    def get_train_dataset(self):
        # should return tf.data.Dataset of tuples (filepath, label)
        raise NotImplementedError('Abstract')

    def get_test_dataset(self):
        # should return tf.data.Dataset of tuples (filepath, label)
        raise NotImplementedError('Abstract')


class Char47K(AbstractDatasetLoader):
    """Loads data from Char47K database.

    The database consists of 3 groups:          (height x width x channels)
        font - images generated from fonts      (128x128x3)
        hand - images drawn on tablet           (900x1200x3)
        img_good - photos of different sizes    (from 6x16x3 to 391x539x3)
        img_bad - photos with worse quality     (from 11x7x3 to 464x325x3)
    It contains 78905 PNG images (font:hand:igood:ibad - 62992:3410:4798:7705).
    To make the number of hand-drawn and real images comparable to font-generated
    ones, it is possible to set images upscaling for training data, which results
    in adding these images 'xxx_upscale' times.
    """
    def __init__(self, hand_upscale=1, images_upscale=1,
            dirs=['font', 'hand', 'img_bad', 'img_good']):
        super().__init__()
        self.hand_up = hand_upscale
        self.images_up = images_upscale
        self.chars74k_dir = os.path.join(self.database_dir, 'chars74k')
        self.dirs = dirs
        self.check()

    def get_train_dataset(self):
        return self._load_files('train')

    def get_test_dataset(self):
        return self._load_files('test')

    @staticmethod
    def path2label(path):
        """Given proper path to a file from the database returns (path, label)."""
        sparse = tf.string_split([path], delimiter='/')
        label = sparse.values[-2]
        numeric_label = tf.where(tf.equal(label, list(LABELS)), name='XDXD')
        numeric_label = tf.squeeze(numeric_label)
        tf.assert_rank(numeric_label, 0)  # should be a scalar
        return path, numeric_label

    def check(self):
        result = subprocess.run(['./prepare_database.py', '--check-only'], cwd=self.chars74k_dir)
        if result.returncode != 0:
            self.logger.warn('Chars74K may be incomplete. Please check in %s.' % self.chars74k_dir)
        else:
            self.logger.debug('Chars74K seems to be complete (all directories exist)')

    def _load_files(self, mode):
        assert mode in ['train', 'test'], 'Wrong mode!'
        # gather file paths
        pattern = '{dir}/{mode}/{label}/*.png'
        patterns = [os.path.join(self.chars74k_dir,
            pattern.format(dir=d, mode=mode, label='*')) for d in self.dirs]
        # get numbers of repeats
        repeats = []
        for dir in self.dirs:
            n = self.hand_up if dir == 'hand' else \
                self.images_up if dir in ['img_bad', 'img_good'] else 1
            repeats.append(n)
        repeats = np.array(repeats, dtype=np.int64)  # can't be int32
        patterns = tf.data.Dataset.from_tensor_slices((patterns, repeats))
        # map each pattern to list of files, then map each file to (file, label)
        dataset = patterns.interleave(lambda pattern, repeats:
            tf.data.Dataset.list_files(pattern, shuffle=mode == 'train') \
            .map(self.path2label).repeat(repeats), cycle_length=1)
        return dataset
