# coding = utf-8
import os

import gc

#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
from time import time

sys.path.append("..")
import pandas as pd
import numpy as np
import pickle
import keras
import keras.backend as K
from keras.models import load_model

from init.config2 import Config2
from my_utils.data_preprocess import to_categorical, word_cnn_train_batch_generator, get_word_seq
from model.deepzoo import get_textcnn
from my_utils.metrics import score
from my_utils.tools import time_cost

print("Load train @ test")
train = pd.read_csv(Config2.train_data_path, sep='\t')
val = pd.read_csv(Config2.test_data_path, sep='\t')
val_label = to_categorical(val.label)

use_model = False

# load_val = True
load_val = False
batch_size = 128
model_name = "word_cnn"
trainable_layer = ["embedding"]
train_batch_generator = word_cnn_train_batch_generator

if load_val:
    val_word_seq = pickle.load(open(Config2.cache_dir + "/g_val_word_seq_%s.pkl" % Config2.word_seq_maxlen, "rb"))
else:
    val_word_seq = get_word_seq(val.content.values)
    gc.collect()
    pickle.dump(val_word_seq, open(Config2.cache_dir + "/g_val_word_seq_%s.pkl" % Config2.word_seq_maxlen, "wb"))
print(np.shape(val_word_seq))

print("Load Word")
word_embed_weight = pickle.load(open(Config2.word_embed_path, "rb"))

if use_model:
    model = load_model(Config2.textcnn_model_path)
    print('load local model %s' % Config2.textcnn_model_path)
else:
    model = get_textcnn(Config2.word_seq_maxlen, word_embed_weight)

start = time()
best_f1 = 0

for i in range(10):

    print('*********Epoch %s/%s Start************' % (i + 1, 10))

    if i == 6:
        K.set_value(model.optimizer.lr, 0.0001)
    if i == 4:
        for l in trainable_layer:
            model.get_layer(l).trainable = True
    model.fit_generator(
        train_batch_generator(train.content.values, train.label.values, batch_size=batch_size),
        epochs=1,
        steps_per_epoch=int(train.shape[0] / batch_size),
        validation_data=(val_word_seq, val_label)
    )
    pred = np.squeeze(model.predict(val_word_seq))
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
        model.save(Config2.cache_dir + '/word_cnn/%s_epoch_%s_%s_%.7s.h5' % (model_name, i, f1, f1_mean))

    time_cost(start, time(), i + 1)

    print('*********Epoch %s/%s End************' % (i + 1, 10))
