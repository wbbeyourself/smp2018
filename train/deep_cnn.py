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

from init.config import Config
from my_utils.data_preprocess import to_categorical, word_cnn_train_batch_generator, get_word_seq
from model.deepzoo import get_textcnn_v2
from my_utils.metrics import score
from my_utils.tools import time_cost

print("Load train @ test")
train = pd.read_csv(Config.train_data_path, sep='\t')
val = pd.read_csv(Config.test_data_path, sep='\t')
val_label = to_categorical(val.label)

load_val = True
batch_size = 128
model_name = "deep_cnn"
trainable_layer = ["embedding"]
train_batch_generator = word_cnn_train_batch_generator

if load_val:
    val_word_seq = pickle.load(open(Config.cache_dir + "/g_val_word_seq_%s.pkl" % Config.word_seq_maxlen, "rb"))
else:
    val_word_seq = get_word_seq(val.content.values)
    gc.collect()
    pickle.dump(val_word_seq, open(Config.cache_dir + "/g_val_word_seq_%s.pkl" % Config.word_seq_maxlen, "wb"))
print(np.shape(val_word_seq))

print("Load Word")
word_embed_weight = pickle.load(open(Config.word_embed_path, "rb"))

model = get_textcnn_v2(Config.word_seq_maxlen, word_embed_weight)
from keras.utils.vis_utils import plot_model

plot_model(model, to_file=Config.cache_dir + '/word_deep_cnn/' + model_name + '.png', show_shapes=True)

best_f1 = 0

start = time()

for i in range(15):

    print('*********Epoch %s/%s Start************' % (i + 1, 15))

    if i == 6:
        K.set_value(model.optimizer.lr, 0.0001)
    if i == 8:
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

    f1_mean = np.mean(f1)

    # set numpy print precision
    np.set_printoptions(precision=5, suppress=True)

    print("precision", pre)
    print("recall", rec)
    print("f1_score", f1)
    print("f1_mean", f1_mean)

    time_cost(start, time(), i + 1)

    print('*********Epoch %s/%s End************' % (i + 1, 15))

    if f1_mean > best_f1:
        best_f1 = f1_mean
        model.save(Config.cache_dir + '/word_deep_cnn/%s_epoch_%s_%s_%.7s.h5' % (model_name, i, f1, best_f1))
