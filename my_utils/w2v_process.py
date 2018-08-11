
# coding: utf-8

# In[1]:


import os
import sys
import time
import pickle
BASE = '/home/wb/smp2018'
sys.path.append(BASE)


# In[2]:


import gensim
import word2vec
import numpy as np
import multiprocessing
from init.config import Config
from collections import defaultdict
from joblib import Parallel, delayed
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from data import *


# In[3]:


vec_dim = 256


# In[4]:


cfg = Config()


# In[5]:


def create_w2vc(overwrite=True):
    if overwrite:
        if os.path.exists(cfg.cache_dir + '/w2v_content_word.txt'):
            os.remove(cfg.cache_dir + '/w2v_content_word.txt')
        if os.path.exists(cfg.cache_dir + '/w2v_content_char.txt'):
            os.remove(cfg.cache_dir + '/w2v_content_char.txt')

        train_data = get_train_all_data()
        vali_data = get_validation_data()

        train_content = train_data["content"]
        vali_content = vali_data["content"]

        print("len of train contents", len(train_content))
        print ("len of vali contents", len(vali_content))
        print("len of total contents:", len(train_content) + len(vali_content))

        def applyParallel(contents, func, n_thread):
            with Parallel(n_jobs=n_thread) as parallel:
                parallel(delayed(func)(c) for c in contents)

        def word_content(content):
            with open(cfg.cache_dir + "/w2v_content_word.txt", "a+") as f:
                f.writelines(content.lower())
                f.writelines('\n')

        def char_content(content):
            with open(cfg.cache_dir + "/w2v_content_char.txt", "a+") as f:
                content = content.lower().replace(" ", "")
                f.writelines(" ".join(content))
                f.writelines("\n")

        applyParallel(train_content, word_content, 25)
        applyParallel(train_content, char_content, 25)
        applyParallel(vali_content, word_content, 25)
        applyParallel(vali_content, char_content, 25)


    # word vector train
    model = gensim.models.Word2Vec(
        LineSentence(cfg.cache_dir + "/w2v_content_word.txt"),
        size=vec_dim,
        window=5,
        min_count=1,
        workers=multiprocessing.cpu_count()
    )
    model.save(cfg.cache_dir + "/content_w2v_word.model")

    # char vector train
    model = gensim.models.Word2Vec(
        LineSentence(cfg.cache_dir + '/w2v_content_char.txt'),
        size=vec_dim,
        window=5,
        min_count=1,
        workers=multiprocessing.cpu_count()
    )
    model.save(cfg.cache_dir + "/content_w2v_char.model")


# - len of train contents 146,341
# - len of vali contents 58,537
# - len of toal contents: 204,878

# In[6]:


# create_w2vc(False)


# In[7]:


def create_word_emb(use_opened=False, overwriter=False):

    vocab = pickle.load(open(cfg.word_vocab_path, 'rb'))
    print(len(vocab))

    if use_opened:
        word_emb = [np.random.uniform(0, 0, 200) for j in range(len(vocab)+1)]
        model = word2vec.load(cfg.open_w2v_path)
    else:
        word_emb = [np.random.uniform(0, 0, 256) for j in range(len(vocab)+1)]
        model = gensim.models.Word2Vec.load(cfg.cache_dir + "/content_w2v_word.model")
    num = 0
    
    for word in vocab:
        index = vocab[word]
        if word in model:
            word_emb[index] = np.array(model[word])
            num += 1
        else:
            word_emb[index] = np.random.uniform(-0.5, 0.5, 200)
    word_emb = np.array(word_emb)
    print("word number: ", num)
    print("vocab size:", len(vocab))
    print("shape of word_emb", np.shape(word_emb))
    if overwriter:
        with open(cfg.word_embed_path, 'wb') as f:
            pickle.dump(word_emb, f)
            print("size of embedding_matrix: ", len(word_emb))
            print("word_embedding finish")


# - size of word embedding_matrix:  649,130

# In[8]:


# create_word_emb(False, True)


# In[9]:


def create_char_emb(overwriter=False):

    vocab = pickle.load(open(cfg.char_vocab_path, 'rb'))
    char_emb = [np.random.uniform(0, 0, 256) for j in range(len(vocab)+1)]
    model = gensim.models.Word2Vec.load(cfg.cache_dir + "/content_w2v_char.model")
    num = 0
    for char in vocab:
        index = vocab[char]
        if char in model:
            char_emb[index] = np.array(model[char])
            num += 1
        else:
            char_emb[index] = np.random.uniform(-0.5, 0.5, 256)
    char_emb = np.array(char_emb)
    print("char number: ", num)
    print("vocab size:", len(vocab))
    print("shape of char_emb", np.shape(char_emb))
    if overwriter:
        with open(Config.char_embed_path, 'wb') as f:
            pickle.dump(char_emb, f)
            print("size of embedding_matrix: ", len(char_emb))
            print("char_embedding finish")


# - size of char embedding_matrix:  7,862

# In[10]:


# create_char_emb(True)


# In[11]:


def create_word_vocab(overwriter=False):
    word_freq = defaultdict(int)

    train_data = get_train_all_data()
    vali_data = get_validation_data()
    train_content = train_data["content"]
    vali_content = vali_data["content"]
    
    t0 = time.time()

    for line in train_content:
        line = line.lower().strip()
        words = line.split(" ")
        for word in words:
            if " " == word or "" == word:
                continue
            word_freq[word] += 1
    
    t1 = time.time()
    print('read train data : %s s' % int(t1-t0))

    for line in vali_content:
        line = line.lower().strip()
        words = line.split(" ")
        for word in words:
            if " " == word or "" == word:
                continue
            word_freq[word] += 1
    
    t2 = time.time()
    print('read vali data : %s s' % int(t2-t0))
    
    vocab = {}
    i = 1
    min_freq = 1
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = i
            i += 1
    vocab['NUM'] = i
    vocab['UNK'] = i+1
    print("size of vocab:", len(vocab))

    if overwriter:
        vocab_file = cfg.cache_dir + '/word_vocab.pk'
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab, f)
        t3 = time.time()
        print("finish to create vocab; cost : %s s" % int(t3-t0))


# - size of word vocab: 649,129

# In[12]:


# create_word_vocab()


# In[13]:


def create_char_vocab(overwriter=False):
    char_freq = defaultdict(int)

    train_data = get_train_all_data()
    vali_data = get_validation_data()
    train_content = train_data["content"]
    vali_content = vali_data["content"]
    
    t0 = time.time()

    for line in train_content:
        line = line.lower().strip()
        line = line.replace(" ", "")
        chars_line = " ".join(line)
        chars = chars_line.split(" ")
        for char in chars:
            if " " == char or "" == char:
                continue
            char_freq[char] += 1
    
    t1 = time.time()
    print('read train data : %s s' % int(t1-t0))

    for line in vali_content:
        line = line.lower().strip()
        line = line.replace(" ", "")
        chars_line = " ".join(line)
        chars = chars_line.split(" ")
        for char in chars:
            if " " == char or "" == char:
                continue
            char_freq[char] += 1
    
    t2 = time.time()
    print('read vali data : %s s' % int(t2-t0))
    
    vocab = {}
    i = 1
    min_freq = 1
    for char, freq in char_freq.items():
        if freq >= min_freq:
            vocab[char] = i
            i += 1
    vocab['NUM'] = i
    vocab['UNK'] = i+1
    print(vocab)
    print("size of vocab:", len(vocab))

    if overwriter:
        vocab_file = Config.cache_dir + '/char_vocab.pk'
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab, f)
        t3 = time.time()
        print("finish to create vocab; cost : %s s" % int(t3-t0))


# - size of char vocab: 7,861

# In[14]:


# create_char_vocab(True)

