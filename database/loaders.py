#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import string
import logging
import subprocess


LABELS = '0123456789' + string.ascii_uppercase + string.ascii_lowercase


class AbstractDatasetLoader:
    """
    Class that creates lists of files for the given dataset (from the datasets'
    directory structure).
    """
    def __init__(self):
        this_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
        self.database_dir = os.path.join(this_dir)
        self.logger = logging.getLogger('dataset_loaders')

    def load_files(self, which):
        raise NotImplementedError("""
        The function to load all pairs (filepath, label) of the given dataset.
        Must return dictionary (with keys from 'which') of tuples of lists:
            { 'test': ([filepath, ...], [label, ...]), 'train': ... }
        which - must be a list of types: ['test', 'train'] (one or both)
        """.strip())


class Char47KLoader(AbstractDatasetLoader):
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
    def __init__(self, hand_upscale=1, images_upscale=1):
        super().__init__()
        self.hand_upscale = hand_upscale
        self.images_upscale = images_upscale
        self.chars74k_dir = os.path.join(self.database_dir, 'chars74k')
        self.main_dirs = ['font', 'hand', 'img_bad', 'img_good']

    def check(self):
        result = subprocess.run(['./prepare_database.py', '--check-only'], cwd=self.chars74k_dir)
        if result.returncode != 0:
            self.logger.warn('Chars74K may be incomplete. Please check in %s.' % self.chars74k_dir)
        else:
            self.logger.debug('Chars74K seems to be complete (all directories exist)')

    def getUpscaling(self, dir):
        if dir == 'hand': return self.hand_upscale
        if dir in ['img_bad', 'img_good']: return self.images_upscale
        return 1

    def load_files(self, which):
        self.check()
        # gather data as tuple of lists ([filepaths], [labels])
        data = {train_or_test: ([], []) for train_or_test in which}
        counts = {tt: {dir: 0 for dir in self.main_dirs} for tt in which}
        for dirname in self.main_dirs:
            upscaling = self.getUpscaling(dirname)
            for train_or_test in which:
                for label in LABELS:
                    classdir_path = os.path.join(self.chars74k_dir, dirname, train_or_test, label)
                    for filename in os.listdir(classdir_path):
                        filepath = os.path.join(classdir_path, filename)
                        numeric_label = LABELS.find(label)
                        repeats = upscaling if train_or_test == 'train' else 1
                        for _ in range(repeats):
                            counts[train_or_test][dirname] += 1
                            data[train_or_test][0].append(filepath)
                            data[train_or_test][1].append(numeric_label)
        for tt in which:
            n = sum(counts[tt][dir] for dir in counts[tt])
            self.logger.info('Loaded %d filenames (%s) from Chars74K' % (n, tt))
            for dir in counts[tt]:
                self.logger.debug('  {0:>{1}}: {2}'.format(dir,
                    max(len(d) for d in self.main_dirs), counts[tt][dir]))
        return data
