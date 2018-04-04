#! /usr/bin/env python3
# coding: utf-8
#
# Downloads and prepares database from:
#   http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

import os
import re
import sys
import argparse
import shutil
import string
import urllib.request
import tarfile
import scipy.io
import numpy as np
from PIL import Image

### Script arguments ###########################################################

parser = argparse.ArgumentParser(description='Prepares Char47K database.')
parser.add_argument('-c', '--check-only', action='store_true',
    help='just checks if all main directories in database exist, returns non-zero if any directory is missing')

### Definitions ################################################################

this_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
base_dir = this_dir
download_dir = os.path.join(base_dir, 'tmp')

urls = [
        'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz',
        'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz',
        'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz',
        #  'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/Lists.tgz'  # not used
    ]
# archive_mappings = {archive_name: [(in_archive_from_dir, to_dir), ...])
archive_mappings = {
    'EnglishFnt.tgz': [('English/Fnt/',             'font/'    ), ],
    'EnglishHnd.tgz': [('English/Hnd/Img/',         'hand/'    ), ],
    'EnglishImg.tgz': [('English/Img/GoodImg/Bmp/', 'img_good/'),
                       ('English/Img/BadImag/Bmp/', 'img_bad/' ), ],
}

# charater classes
char47k_class_numbers = np.arange(1, 62+1)
classes = '0123456789' + string.ascii_uppercase + string.ascii_lowercase
assert len(classes) == len(char47k_class_numbers)

# for spliting samples into training/test sets "deterministically randomly" - random-like but each time the same
fixed_pseudorandom_seed = 135797531
train_samples_percentage = 80

### Functions ##################################################################

def maybe_download():
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    print('Trying to download files...')
    for url in urls:
        name = url.split('/')[-1]
        filepath = os.path.join(download_dir, name)
        print('  ... %s ...' % name, end='', flush=True)
        if os.path.exists(filepath):
            print(' exists')
        else:
            print(' downloading ...', end='', flush=True)
            urllib.request.urlretrieve(url, filepath)
            print(' done')

def assert_tarfile(tar):
    # whatever, just check if the archive is safe
    assert all(not (name.startswith('/') or name.startswith('..')) for name in tar.getnames()), 'Dangerous tarfile?!'

def extract_samples(tar, tar_fromdir, destdir, print_base_str):
    # tar_fromdir must be a path to the directory that consists only of direcotries SampleXXX with images
    # filter only files from tar_fromdir, remove all temporary *~ files, remove non-files
    tar_members = filter(lambda member: member.path.startswith(tar_fromdir), tar.getmembers())
    tar_members = filter(lambda member: not member.path.endswith('~'), tar_members)
    tar_members = filter(lambda member: member.isfile(), tar_members)
    tar_members = list(tar_members)
    # split files into classes and alter paths to remove preceiding directories
    #  and verbosely name classes' directories
    class_members = {class_name: [] for class_name in classes}
    pattern = re.compile(r'Sample([0-9]{3})')
    for member in tar_members:
        member.path = member.path[len(tar_fromdir):]
        match = pattern.search(member.path)
        if match:
            class_n = int(match.groups()[0])
            new_class = classes[class_n - 1]
            member.path = member.path[:match.start()] + new_class + member.path[match.end():]
            class_members[new_class].append(member)
    # class_members has structure {class: [all, image, files(TarInfo), from, that, class, ...]}
    # split pseudo-randomly to train/test sets
    # using fixed seed, so it should give the same results each time
    np.random.seed(fixed_pseudorandom_seed)
    train_members, test_members = [], []
    for classname in class_members.keys():
        np.random.shuffle(class_members[classname])
        n_training = int(train_samples_percentage/100 * len(class_members[classname]))
        train_members.extend(class_members[classname][:n_training])
        test_members.extend(class_members[classname][n_training:])
    # extract files, doing it sequentially is MUCH faster (at least on HDD)
    n_all = len(train_members) + len(test_members)
    n_cur = 0
    template = '\r%s %{}d/%{}d'.format(len(str(n_all)), len(str(n_all)))
    print_info = lambda n: print(template % (print_base_str, n, n_all), end='')
    print_info(n_cur)
    for member in tar.getmembers():
        if member in train_members:
            tar.extract(member, path=os.path.join(destdir, 'train'))
        elif member in test_members:
            tar.extract(member, path=os.path.join(destdir, 'test'))
        else:
            continue
        n_cur += 1
        print_info(n_cur)
    last_string = template % (print_base_str, n_cur, n_all)
    return last_string

def maybe_unarchive():
    print('Extracting archives...', flush=True)
    for archive_name, mappings in archive_mappings.items():
        base = '  ... %s' % archive_name
        print('%s ... opening' % base, end='', flush=True)
        tar = tarfile.open(os.path.join(download_dir, archive_name))
        assert_tarfile(tar)
        base = '%s ... extracting ... ' % base
        print('\r' + base, end='', flush=True)
        for from_dir, to_dir in mappings:
            if os.path.exists(to_dir):
                base += 'exists ... '
                print('\r' + base, end='', flush=True)
                continue
            last_string = extract_samples(tar, from_dir, os.path.join(base_dir, to_dir), print_base_str=base)
            base = last_string + ' ... '
            print('\r' + base, end='', flush=True)
        print('done', flush=True)

def maybe_resize():
    # resize hand images from 1200x900 to 120x90 because resizing them each time
    # is too expensive
    destsize = (120, 90)
    filepaths = []
    print('Resizing...')
    print('  ... listing files ...', flush=True)
    for from_dir, to_dir in archive_mappings['EnglishHnd.tgz']:
        base_path = os.path.join(base_dir, to_dir)
        for type in ['train', 'test']:
            base_path_2 = os.path.join(base_path, type)
            for label_dir in os.listdir(base_path_2):
                base_path_3 = os.path.join(base_path_2, label_dir)
                pngs = [os.path.join(base_path_3, fname)
                    for fname in os.listdir(base_path_3) if fname.endswith('.png')]
                filepaths.extend(pngs)
    i, n = 0, len(filepaths)
    print('\r  ... resizing images ... %d/%d' % (i, n), flush=True, end='')
    for filepath in filepaths:
        with Image.open(filepath) as img:
            # check the size situation
            is_original_size = img.size == (1200, 900)
            if not is_original_size:
                if img.size == destsize:  # is already resized
                    i += 1
                    continue
                # else something completely wrong
                print('ERROR: database polluted: one of files had neither original size, nor the desired size!',
                    file=sys.stderr)
                return
            assert img.size == (1200, 900), 'All images from EnglishHnd should have size 1200x900'
            img_new = img.resize(destsize, Image.BILINEAR)
        img_new.save(filepath)
        print('\r  ... resizing images ... %d/%d' % (i, n), end='')
        i += 1
    print('\r  ... resizing images ... %d/%d ... done' % (i, n), flush=True)



### Main #######################################################################

if __name__ == '__main__':
    args = parser.parse_args()

    destdirs = [mapping[1] for mappings in archive_mappings.values() for mapping in mappings]
    all_dirs_exist = all(dirname.strip('/') in os.listdir(base_dir) for dirname in destdirs)

    if args.check_only:
        sys.exit(0 if all_dirs_exist else 1)

    if all_dirs_exist:
        print('All directories exist. If you want fresh database, remove them first.')
    else:
        print('No database or missing a directory.')
        answer = input('Starting whole database preparation, proceed? [y/N] ')
        if not answer.lower().strip() in ['y', 'yes']:
            print('Aborting')
            sys.exit()

        maybe_download()
        maybe_unarchive()
        maybe_resize()

    if os.path.exists(download_dir):
        answer = input('Do you want to remove temporary files? [y/N] ')
        if not answer.lower().strip() in ['y', 'yes']:
            print('Aborting')
            sys.exit()
        shutil.rmtree(download_dir)
