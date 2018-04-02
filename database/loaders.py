#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import subprocess

class AbstractDatasetLoader:
    """
    Class that creates lists of files for the given dataset (from the datasets'
    directory structure).
    """
    def __init__(self, labels):
        this_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
        self.database_dir = os.path.join(this_dir)
        self.logger = logging.getLogger('dataset_loaders')
        self.labels = labels

    def load_files(self, which):
        raise NotImplementedError("""
        The function to load all pairs (filepath, label) of the given dataset.
        Must return dictionary (with keys from 'which') of tuples of lists:
            { 'test': ([filepath, ...], [label, ...]), 'train': ... }
        which - must be a list of types: ['test', 'train'] (one or both)
        """.strip())

class Char47KLoader(AbstractDatasetLoader):
    def load_files(self, which):
        """Loads data from Char47K database.

        It contains 78905 PNG images (font:hand:igood:ibad - 62992:3410:4798:7705).
        The database consists of 3 groups:          (height x width x channels)
            font - images generated from fonts      (128x128x3)
            hand - images drawn on tablet           (900x1200x3)
            img_good - photos of different sizes    (from 6x16x3 to 391x539x3)
            img_bad - photos with worse quality     (from 11x7x3 to 464x325x3)
        """
        chars74k_dir = os.path.join(self.database_dir, 'chars74k')
        result = subprocess.run(['./prepare_database.py', '--check-only'], cwd=chars74k_dir)
        if result.returncode != 0:
            self.logger.warn('Chars74K may be incomplete. Please check in %s.' % chars74k_dir)
        # gather data as tuple of lists ([filepaths], [labels])
        data = {train_or_test: ([], []) for train_or_test in which}
        for dirname in ['font/', 'hand/', 'img_bad', 'img_good']:
            for train_or_test in which:
                for label in self.labels:
                    classdir_path = os.path.join(chars74k_dir, dirname, train_or_test, label)
                    for filename in os.listdir(classdir_path):
                        filepath = os.path.join(classdir_path, filename)
                        numeric_label = self.labels.find(label)
                        data[train_or_test][0].append(filepath)
                        data[train_or_test][1].append(numeric_label)
        return data
