# coding=utf-8

import os

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import pickle
import keras
import keras.backend as K

from init.config import Config
from my_utils.data_preprocess import to_categorical, word_char_cnn_train_batch_generator
from model.deepzoo import get_word_cgru_char_rnn
from my_utils.metrics import score

print("Load Train && Val")
train = pd.read_csv(Config.train_data_path, sep='\t')
val = pd.read_csv(Config.test_data_path, sep='\t')
val_label = to_categorical(val.label)

load_val = True
batch_size = 128
model_name = "word_cgru_char_rnn"
trainable_layer = ["word_embedding", "char_embedding"]
train_batch_generator = word_char_cnn_train_batch_generator

print("Load Val Data")
val_word_seq = pickle.load(open(Config.cache_dir + "/g_val_word_seq_%s.pkl"%Config.word_seq_maxlen, "rb"))
val_char_seq = pickle.load(open(Config.cache_dir + "/g_val_char_seq_%s.pkl"%Config.char_seq_maxlen, "rb"))
val_seq = [val_word_seq, val_char_seq]

print("Load Word && Char Embed")
word_embed_weight = pickle.load(open(Config.word_embed_path, "rb"))
char_embed_weight = pickle.load(open(Config.char_embed_path, "rb"))

model = get_word_cgru_char_rnn(Config.word_seq_maxlen, Config.char_seq_maxlen, word_embed_weight, char_embed_weight)
from keras.utils.vis_utils import plot_model
plot_model(model, to_file= model_name +'.png',show_shapes=True)

for i in range(15):
    if i == 6:
        K.set_value(model.optimizer.lr, 0.0001)
    if i == 9:
        for l in trainable_layer:
            model.get_layer(l).trainable = True
    model.fit_generator(
        train_batch_generator(train.content.values, train.label.values, batch_size=batch_size),
        epochs=1,
        steps_per_epoch= int(train.shape[0]/ batch_size),
        validation_data = (val_seq, val_label)
    )
    pred = np.squeeze(model.predict(val_seq))
    pre, rec, f1 = score(pred, val_label)
    print("precision", pre)
    print("recall", rec)
    print("f1_score", f1)
    model.save(Config.cache_dir + "/word_cgru_char_rnn/dp_embed_%s_epoch_%s_%s.h5"%(model_name, i, f1))