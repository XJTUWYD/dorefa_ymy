#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: age_gender.py
# Author: Keisuke Yamaya <yamaya@val.cs.tut.ac.jp>

import os, sys
import pickle as pkl
import numpy as np
import random
import six
from six.moves import urllib, range
import copy
import logging
import cv2

from ...utils import logger, get_rng, get_dataset_path
from ...utils.fs import download
from ..base import RNGDataFlow

__all__ = ['Age', 'Gender']
NUM_CLASSES = 8 # 分類するクラス数
IMG_SIZE = 256 # 画像の1辺の長さ
age_dict = {
    "(0, 2)":0,
    "(4, 6)":1,
    "(8, 12)":2,
    "(15, 20)":3,
    "(25, 32)":4,
    "(38, 43)":5,
    "(48, 53)":6,
    "(60, 100)":7,
}

# Adience benchmark データセットを読む
def read_adience(filenames, age_or_gender, train_or_test):
    assert age_or_gender == 'age' or age_or_gender == 'gender'
    ret = []
    age_list = get_age_list(age_or_gender, train_or_test)
    idx = 0
    for fname in filenames:
        img = cv2.imread(fname)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # 画素値を0-1にして、1列のvectorにする
        img = img.flatten().astype(np.float32) / 255.0
        # one_hot_vectorなラベルを追加
        label = np.zeros(NUM_CLASSES)
        label[age_dict[age_list[idx]]] = 1
        ret.append([img, label])
        idx += 1
    return ret

def get_age_list(age_or_gender, train_or_test):
    if age_or_gender == 'age':
        if train_or_test == 'train':
            with open('/home/ubuntu/yamaya/AgeGender/age_train_list.dump', 'r') as f1:
                age_list = pkl.load(f1)
        else:
            with open('/home/ubuntu/yamaya/AgeGender/age_test_list.dump', 'r') as f2:
                age_list = pkl.load(f2)
    else:
        if train_or_test == 'train':
            with open('/home/ubuntu/yamaya/AgeGender/gender_train_list.dump', 'r') as f3:
                age_list = pkl.load(f3)
        else:
            with open('/home/ubuntu/yamaya/AgeGender/gender_train_list.dump', 'r') as f4:
                age_list = pkl.load(f4)
    return age_list

def get_filenames(age_or_gender, train_or_test):
    # benchmark/faces/7153718@N04/coarse_tilt_aligned_face....jpg
    assert age_or_gender == 'age' or age_or_gender == 'gender'
    if age_or_gender == 'age':
        if train_or_test == 'train':
            with open('/home/ubuntu/yamaya/AgeGender/age_train_fnames.dump', 'r') as f1:
                filenames = pkl.load(f1)
        else:
            with open('/home/ubuntu/yamaya/AgeGender/age_test_fnames.dump', 'r') as f2:
                filenames = pkl.load(f2)
    elif age_or_gender == 'gender':
        if train_or_test == 'train':
            with open('/home/ubuntu/yamaya/AgeGender/gender_train_fnames.dump', 'r') as f3:
                filenames = pickle.load(f3)
        else:
            with open('/home/ubuntu/yamaya/AgeGender/gender_test_fnames.dump', 'r') as f4:
                filenames = pkl.load(f4)
    return filenames

class AgeAndGenderBase(RNGDataFlow):
    """
    Return [image, label],
        image is 32x32x3 in the range [0,255]
    """
    def __init__(self, train_or_test, shuffle=True, dir=None, age_or_gender='age'):
        """
        Args:
            train_or_test: string either 'train' or 'test'
            shuffle: default to True
            age_or_gender: string either 'age' or 'gender'
        """
        assert train_or_test in ['train', 'test']
        assert age_or_gender == 'age' or age_or_gender == 'gender'
        if dir is None:
            dir = './'
        self.age_or_gender = age_or_gender
        fnames = get_filenames(age_or_gender, train_or_test)
        self.fs = fnames
        for f in self.fs:
            if not os.path.isfile(f):
                raise ValueError('Failed to find file: ' + f)
        self.train_or_test = train_or_test
        self.data = read_adience(self.fs, age_or_gender, train_or_test)
        self.dir = dir
        self.shuffle = shuffle

    # trainとtestは数えるとこれだけ
    def size(self):
        return 13717 if self.train_or_test == 'train' else 3676

    def get_data(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield copy.copy(self.data[k])

    def get_per_pixel_mean(self):
        """
        return a mean image of all (train and test) images of size 32x32x3
        """
        fnames = get_filenames(self.dir, self.age_or_gender)
        all_imgs = [x[0] for x in read_adience(fnames, self.age_or_gender)]
        arr = np.array(all_imgs, dtype='float32')
        mean = np.mean(arr, axis=0)
        return mean

    def get_per_channel_mean(self):
        """
        return three values as mean of each channel
        """
        mean = self.get_per_pixel_mean()
        return np.mean(mean, axis=(0,1))

class Age(AgeAndGenderBase):
    def __init__(self, train_or_test, shuffle=False, dir=None):
        super(Age, self).__init__(train_or_test, shuffle, dir, 'age')

class Gender(AgeAndGenderBase):
    def __init__(self, train_or_test, shuffle=True, dir=None):
        super(Gender, self).__init__(train_or_test, shuffle, dir, 'gender')

if __name__ == '__main__':
    ds = Cifar10('train')
    print(ds)
    #from tensorpack.dataflow.dftools import dump_dataset_images
    #mean = ds.get_per_channel_mean()
    #print(mean)
    #dump_dataset_images(ds, '/tmp/cifar', 100)

    #for (img, label) in ds.get_data():
        #from IPython import embed; embed()
        #break
