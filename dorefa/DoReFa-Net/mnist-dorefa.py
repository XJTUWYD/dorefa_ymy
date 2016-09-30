#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-convnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import tensorflow as tf
import os, sys
import argparse

import cv2
import argparse
import numpy as np
import multiprocessing
import msgpack

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from dorefa import get_dorefa
import sys
# モジュール(もしくはそのフォルダ)へのパスを追加
#sys.path.append('/home/tomohiro/github/tensorpack/module/')
#from . import imgaug

"""
MNIST ConvNet example.
about 0.6% validation error after 30 epochs.
"""

IMAGE_SIZE = 28

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputVar(tf.int32, (None,), 'label') ]

    def _build_graph(self, input_vars, is_training):
        is_training = bool(is_training)
        keep_prob = tf.constant(0.5 if is_training else 1.0)


        image, label = input_vars
        image = tf.expand_dims(image, 3)    # add a single channel


        fw, fa, fg = get_dorefa(1, 2, 7)
        # monkey-patch tf.get_variable to apply fw
        old_get_variable = tf.get_variable #  weightの更新

        nl = PReLU.f
        image = image * 2 - 1

        def new_get_variable(name, shape=None, **kwargs):
            v = old_get_variable(name, shape, **kwargs)
            # don't binarize first and last layer
            if name != 'W' or 'conv0' in v.op.name or 'fct' in v.op.name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)
        tf.get_variable = new_get_variable

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)    # still use relu for 32bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def cabs(x):
            return tf.minimum(1.0, tf.abs(x), name='cabs')

        def activate(x):
            return fa(cabs(x))  # 活性化関数の出力のクリップ


        with argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
            argscope(Conv2D, kernel_shape=3, nl=nl, out_channel=32):
            logits = (LinearWrap(image) # the starting brace is only for line-breaking
                    .Conv2D('conv0', padding='VALID')#.apply(fg).BatchNorm('bn1',use_local_stat=is_training)
                    .MaxPooling('pool0', 2)
                    .apply(activate)

                    .Conv2D('conv1', padding='SAME')
                    .apply(fg)
                    .BatchNorm('bn2')
                    .apply(activate)

                    .Conv2D('conv2', padding='VALID')
                    .apply(fg)
                    .BatchNorm('bn3')
                    .MaxPooling('pool1', 2)
                    .apply(activate)

                    .Conv2D('conv3', padding='VALID')
                    .apply(fg)
                    .BatchNorm('bn4')
                    .apply(activate)

                    .FullyConnected('fc0', 512)
                    .apply(fg)
                    .BatchNorm('bn5')#.tf.nn.dropout(keep_prob)
                    .apply(cabs)
                    .FullyConnected('fc1', out_dim=10, nl=tf.identity)())

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = symbolic_functions.prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        summary.add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(1e-5,
                         regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        summary.add_moving_summary(cost, wd_cost)

        summary.add_param_summary([('.*/W', ['histogram'])])   # monitor histogram of all W
        self.cost = tf.add_n([wd_cost, cost], name='cost')

def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    return train, test

def run_test(path, input):
    #/home/tomohiro/github/tensorpack/tensorpack/dataflow/dataset/mnist_data/t10k-images.idx3-ubyte
    #/home/tomohiro/github/tensorpack/examples/DoReFa-Net/train_log/mnist-dorefa0831-110142/model-1404
    #param_dict = np.load(path).item()

    pred_config = PredictConfig(
        model=Model(),
        input_var_names=['input'],
        session_init=SaverRestore(path),
        session_config=get_default_sess_config(0.9),
        output_var_names=['output']   # output:0 is the probability distribution
    )
    predict_func = get_predict_func(pred_config)

    import cv2
    im = cv2.imread(input)
    assert im is not None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.reshape(im, (1, 28, 28, 3)).astype('float32')
    im = im - 110
    outputs = predict_func([im])[0]
    prob = outputs[0]
    ret = prob.argsort()[-10:][::-1]
    print (ret)

    #meta = ILSVRCMeta().get_synset_words_1000()
    #print [meta[k] for k in ret]

def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        session_config=get_default_sess_config(0.9),
        input_var_names=['input'],
        output_var_names=['output']
    )
    predict_func = get_predict_func(pred_config)
    #meta = dataset.ILSVRCMeta()
    #pp_mean = meta.get_per_pixel_mean()
    #pp_mean_224 = pp_mean[16:-16,16:-16,:]#114->28
    #words = meta.get_synset_words_1000()

    def resize_func(im):
        h, w = im.shape[:2]
        scale = 256.0 / min(h, w)
        desSize = map(int, (max(28, min(w, scale * w)),\
                            max(28, min(h, scale * h))))
        im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
        return im
    transformers = imgaug.AugmentorList([
        imgaug.MapImage(resize_func),
        imgaug.CenterCrop((28, 28)),
        imgaug.MapImage(lambda x: x - 0),
    ])

    for f in inputs:#入力画像の数
        assert os.path.isfile(f)
        img = cv2.imread(f).astype('float32')
        assert img is not None
        new_img=1*[28*[28*[0]]]
        for i in xrange(28):
            for j in xrange(28):
                new_img[0][j][j] = img[j][i][0]
        #img = transformers.augment(img)[:,:,:]#ここがエラーの原因
        outputs = predict_func([new_img])[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]

        #names = [i for i in ret]
        #print(f + ":")
        #print(list(zip(names, prob[ret])))
        print (prob[ret])

def get_config():
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data()
    step_per_epoch = dataset_train.size()

    lr = tf.train.exponential_decay(
        learning_rate=1e-3,
        global_step=get_global_step_var(),
        decay_steps=dataset_train.size() * 10,
        decay_rate=0.3, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            InferenceRunner(dataset_test,
                [ScalarStats('cost'), ClassificationError() ]),
        ]),
        session_config=get_default_sess_config(0.5),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=100,
    )
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a checkpoint, or a npy (given as the pretrained model)')


    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--dorefa',
            help='number of bits for W,A,G, separated by comma. Defaults to \'1,2,4\'',
            default='1,2,4')
    parser.add_argument('--run', help='run on a list of images with the pretrained model', nargs='*')
    #parser.add_argument('--input', help='an input image', required=True)
    #parser.add_argument('--input', help='an input image', required=True)
    args = parser.parse_args()

    BITW, BITA, BITG = map(int, args.dorefa.split(','))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)#args.load
    if args.run:
        #run_image(Model(), ParamRestore(np.load(args.load, encoding='latin1').item()), args.run)

        run_image(Model(), SaverRestore(args.load), args.run)
        pass
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    #SimpleTrainer(config).train_test()
    SimpleTrainer(config).train()



    #SimpleTrainer(config).run_step()
    #trainer=QueueInputTrainer(config)
    #config.callbacks._after_train()
    #SyncMultiGPUTrainer(config).train()
    #run_image(Model(), ParamRestore(np.load(args.load, encoding='latin1').item()), args.run)

    #run_image(Model(), SaverRestore(args.load), args.run)
    #run_test(args.load, args.input)
    #run_test("/home/tomohiro/github/tensorpack/examples/DoReFa-Net/train_log/mnist-dorefa0831-110142/model-1404","/home/tomohiro/github/tensorpack/tensorpack/dataflow/dataset/mnist_data/t10k-images.idx3-ubyte")
    #どこにvalidation errorなどのprint文があるか調査する。
