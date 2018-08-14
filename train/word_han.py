# coding=utf-8

import os
import sys

sys.path.append("..")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import pandas as pd
import numpy as np
import keras
import keras.backend as K
import pickle
import gc

from time import time
from init.config import Config
from my_utils.data_preprocess import to_categorical, word_han_train_batch_generator, word_han_preprocess
from model.deepzoo import get_han
from my_utils.metrics import score
from my_utils.tools import time_cost

load_val = True
# load_val = False
batch_size = 128
model_name = "word_han"
trainable_layer = ["embeding"]
train_batch_generator = word_han_train_batch_generator

print("Load train @ test")
train = pd.read_csv(Config.train_data_path, sep='\t')
val = pd.read_csv(Config.test_data_path, sep='\t')
val_label = to_categorical(val.label)
print("Load ok")

if load_val:
    val_han_word_seq = pickle.load(
        open(Config.cache_dir + "/g_val_han_word_seq_%s.pkl" % (Config.sentence_num * Config.sentence_word_length),
             "rb"))
else:
    val_han_word_seq = word_han_preprocess(val.content.values)
    gc.collect()
    pickle.dump(val_han_word_seq, open(
        Config.cache_dir + '/g_val_han_word_seq_%s.pkl' % (Config.sentence_num * Config.sentence_word_length), "wb"))
print(np.shape(val_han_word_seq))

print("Load Word")
word_embed_weight = pickle.load(open(Config.word_embed_path, "rb"))
print("Load ok")

model = get_han(Config.sentence_num, Config.sentence_word_length, word_embed_weight)
from keras.utils.vis_utils import plot_model

plot_model(model, to_file=Config.cache_dir + '/word_han/' + model_name + '.png', show_shapes=True)

start = time()
best_f1 = 0

for i in range(12):

    print('*********Epoch %s/%s Start************' % (i + 1, 12))

    if i == 5:
        K.set_value(model.optimizer.lr, 0.0001)
    if i == 6:
        for l in trainable_layer:
            model.get_layer(l).trainable = True
    model.fit_generator(
        train_batch_generator(train.content.values, train.label.values, batch_size=batch_size),
        epochs=1,
        steps_per_epoch=int(train.shape[0] / batch_size),
        validation_data=(val_han_word_seq, val_label)
    )
    pred = np.squeeze(model.predict(val_han_word_seq))
    pre, rec, f1 = score(pred, val_label)

    print("precision", pre)
    print("recall", rec)
    print("f1_score", f1)

    f1_mean = np.mean(f1)

    # set numpy print precision
    np.set_printoptions(precision=5, suppress=True)

    print("f1_mean", f1_mean)

    if f1_mean > best_f1:
        best_f1 = f1_mean
        model.save(Config.cache_dir + '/word_han/%s_epoch_%s_%s_%.7s.h5' % (model_name, i, f1, f1_mean))

    time_cost(start, time(), i + 1)

    print('*********Epoch %s/%s End************' % (i + 1, 12))
