#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: age_gender_convnet.py
# Author: Keisuke YAMAYA <yamaya@val.cs.tut.ac.jp>

import tensorflow as tf
import argparse
import numpy as np
import os
import pickle as pkl
import glob
import cv2

from tensorpack import *
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.summary import *

"""
Age Gender CNN のベースライン実装
"""

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
CLASS_NUM = 8

class Model(ModelDesc):

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 256, 256, 3], 'input'),
                InputVar(tf.int32, [None, CLASS_NUM], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        is_training = get_current_tower_context().is_training
        keep_prob = tf.constant(0.5 if is_training else 1.0) # ドロップアウトする率

        if is_training:
            tf.image_summary("train_image", image, 10)

        image = image / 4.0     # just to make range smaller
        with argscope(Conv2D, nl=BNReLU(), use_bias=False, kernel_shape=3):
            logits = LinearWrap(image) \
                    .Conv2D('conv1', out_channel=96, stride=4, kernel_shape=7) \
                    .tf.nn.relu(name='relu2') \
                    .MaxPooling('pool1', 3, stride=2) \
                    .tf.nn.local_response_normalization(depth_radius=5, alpha=0.0001, beta=0.75, name='norm1') \
                    .Conv2D('conv2', out_channel=256, kernel_shape=5) \
                    .tf.nn.relu('relu2') \
                    .MaxPooling('pool2', 3, stride=2) \
                    .tf.nn.local_response_normalization(alpha=0.0001, beta=0.75, name='norm2') \
                    .Conv2D('conv3', out_channel=384, kernel_shape=3) \
                    .tf.nn.relu(name='relu3') \
                    .MaxPooling('pool5', 3, stride=2) \
                    .FullyConnected('fc6', 512) \
                    .tf.nn.relu(name='relu6') \
                    .tf.nn.dropout(keep_prob) \
                    .FullyConnected('fc7', 512) \
                    .tf.nn.relu(name='relu7') \
                    .tf.nn.dropout(keep_prob) \
                    .FullyConnected('fc8', out_dim=8, nl=tf.identity)()
        prob = tf.nn.softmax(logits, name='prob')

        #cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.nn.softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = symbf.prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(0.004,
                         regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

# 年齢推定のために、画像と年齢のラベルを読む
def get_age_data(train_or_test):
    NUM_CLASSES = 8 # 分類するクラス数
    IMG_SIZE = 128 # 画像の1辺の長さ

    isTrain = train_or_test == 'train'
    if isTrain:
        # 学習画像データ
        train_image = []
        # 学習データのラベル
        train_label = []
        # 保存用のファイル名リスト
        fnames = []
        ageList = []
        # data0からdata3までの4つを学習データとして読み込む
        for idx in range(0, 4):
            f = open('./benchmark/data' + str(idx) + '.txt', 'r')
            line = f.readline()
            while line:
                splitted = line[:-1].split('\t') # 改行コードを取り除いてスペースでスプリット
                dirName = splitted[0]
                fileName = splitted[1]
                age = str(splitted[2])
                if age_dict.get(age) is None:
                    line = f.readline()
                    continue
                # いま見ているディレクトリから、fileNameを部分文字列として含むようなファイルを見つける
                fileList = glob.glob('./benchmark/faces/' + dirName + '/*.jpg')
                ret = -1
                idx = 0 # fileListの何番目にあったか
                for fl in fileList:
                    ret = fl.find(fileName)
                    if ret != -1:
                        break
                    idx += 1
                fnames.append(fileList[idx])
                img = cv2.imread(fileList[idx]) # 画像読み込み
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # IMG_SIZE*IMG_SIZEにリサイズ
                # 画素値を0-1にして、1列のvectorにする
                img = img.flatten().astype(np.float32) / 255.0
                train_image.append(img)
                # one_hot_vectorなラベルを追加
                tmp = np.zeros(NUM_CLASSES)
                tmp[age_dict[age]] = 1
                train_label.append(tmp)
                ageList.append(age)
                line = f.readline()

        f.close()
        # numpy配列に変換
        train_image = np.asarray(train_image)
        train_label = np.asarray(train_label)
        # [image, label], ... となるように結合する
        ds = []
        for idx in range(0, len(train_image)):
            tmp = [train_image[idx], train_label[idx]]
            ds.append(tmp)
        with open('train.dump', 'w') as fp:
            pkl.dump(ds, fp)
        with open('age_train_fnames.dump', 'w') as fp2:
            pkl.dump(fnames, fp2)
        with open('age_train_list.dump', 'w') as fp3:
            pkl.dump(ageList, fp3)
        print('train ds was saved. train size is ' + str(len(train_image)))
        return ds
    else:
        # テスト画像データ
        test_image = []
        # テストデータのラベル
        test_label = []
        # 保存用のファイル名リスト
        fnames = []
        ageList = []
        # data4をテストデータとして読み込む
        idx = 4
        f = open('./benchmark/data' + str(idx) + '.txt', 'r')
        line = f.readline()
        while line:
            splitted = line[:-1].split('\t') # 改行コードを取り除いてスペースでスプリット
            dirName = splitted[0]
            fileName = splitted[1]
            age = str(splitted[2])
            if age_dict.get(age) is None:
                line = f.readline()
                continue
            # いま見ているディレクトリから、fileNameを部分文字列として含むようなファイルを見つける
            fileList = glob.glob('./benchmark/faces/' + dirName + '/*.jpg')
            ret = -1
            idx = 0 # fileListの何番目にあったか
            for fl in fileList:
                ret = fl.find(fileName)
                if ret != -1:
                    break
                idx += 1
            fnames.append(fileList[idx])
            img = cv2.imread(fileList[idx]) # 画像読み込み
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # IMG_SIZE*IMG_SIZEにリサイズ
            # 画素値を0-1にして、1列のvectorにする
            img = img.flatten().astype(np.float32) / 255.0

            # one_hot_vectorなラベルを追加
            tmp = np.zeros(NUM_CLASSES)
            #print(fileName)
            test_image.append(img)
            tmp[age_dict[age]] = 1
            test_label.append(tmp)
            ageList.append(age)
            line = f.readline()

        f.close()
        # numpy配列に変換
        test_image = np.asarray(test_image)
        test_label = np.asarray(test_label)
        # [image, label], ... となるように結合する
        ds = []
        for idx in range(0, len(test_image)):
            tmp = [test_image[idx], test_label[idx]]
            ds.append(tmp)
        with open('test.dump', 'w') as fp:
            pkl.dump(ds, fp)
        with open('age_test_fnames.dump', 'w') as fp2:
            pkl.dump(fnames, fp2)
        with open('age_test_list.dump', 'w') as fp3:
            pkl.dump(ageList, fp3)
        print('test saved. test size is ' + str(len(test_image)))
        return ds

# 性別推定のために、画像と性別のラベルを読む
def get_gender_data():
    return

def get_data(train_or_test, age_or_gender):
    isTrain = train_or_test == 'train'
    isAge = age_or_gender == 'age'
    if isAge:
        #ds = get_age_data(train_or_test)
        ds = dataset.Age(train_or_test)
    else:
        #ds = get_gender_data(train_or_test)
        ds = dataset.Gender(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.RandomCrop((256, 256)),
            imgaug.Flip(horiz=True),
            imgaug.Brightness(63),
            imgaug.Contrast((0.2,1.8)),
            imgaug.GaussianDeform(
                [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],
                (256,256), 0.2, 3),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.CenterCrop((256, 256)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

# age_or_gender: 'age' or 'gender'の文字列
def get_config(age_or_gender):
    logger.auto_set_dir()

    # データセットの用意
    dataset_test = get_data('test', age_or_gender)
    dataset_train = get_data('train', age_or_gender)
    step_per_epoch = dataset_train.size()

    sess_config = get_default_sess_config(0.5)

    nr_gpu = get_nr_gpu()
    lr = tf.train.exponential_decay(
        learning_rate=1e-2,
        global_step=get_global_step_var(),
        decay_steps=step_per_epoch * (30 if nr_gpu == 1 else 20),
        decay_rate=0.5, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            InferenceRunner(dataset_test, ClassificationError())
        ]),
        session_config=sess_config,
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=100,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--ag', help='age or gender', default='age') # ageかgenderか指定
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with tf.Graph().as_default():
        config = get_config(args.ag)
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        SimpleTrainer(config).train()
        #AsyncMultiGPUTrainer(config).train()
