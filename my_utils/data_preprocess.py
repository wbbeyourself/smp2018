
# coding: utf-8

# In[1]:


import sys
BASE = '/home/wb/smp2018'
sys.path.append(BASE)


# In[2]:


import re
import pickle
from data import *
import numpy as np
from init.config import Config
from keras.preprocessing import sequence


# In[3]:


cfg = Config()


# In[4]:


word_vocab = pickle.load(open(cfg.word_vocab_path, 'rb'))
char_vocab = pickle.load(open(cfg.char_vocab_path, 'rb'))


# In[5]:


def convert_num(word):
    pattern = re.compile('[0-9]+')
    match = pattern.findall(word)
    if match:
        return True
    else:
        return False


# In[6]:


def word_han_preprocess(contents, sentence_num=cfg.sentence_num, sentence_length=cfg.sentence_word_length, keep=False):
    contents_seq = np.zeros(shape=(len(contents), sentence_num, sentence_length))
    
    total = len(contents)
    one = total // 100
    
    for index, content in enumerate(contents):
        sentences = SentenceSplitter.split(content)
        word_seq = get_word_seq(sentences, word_maxlen=sentence_length)
        word_seq = word_seq[:sentence_num]
        contents_seq[index][:len(word_seq)] = word_seq
        
        if index % one == 0:
            print('word_han_preprocess %s %%' % str((index * 100) // total))
            
    return contents_seq


# In[7]:


def get_word_seq(contents, word_maxlen=cfg.word_seq_maxlen, mode="post", keep=False, verbost=False):
    unknow_index =len(word_vocab)
    word_r = []
    for content in contents:
        word_c = []
        content = content.lower().strip()
        words = content.split(" ")
        for word in words:
            if convert_num(word):
                word = 'NUM'
            if word in word_vocab:
                index = word_vocab[word]
            else:
                index = unknow_index
            word_c.append(index)
        word_c = np.array(word_c)
        word_r.append(word_c)
    word_seq = sequence.pad_sequences(word_r, maxlen=word_maxlen, padding=mode, truncating=mode, value=0)
    return word_seq


# In[18]:


def get_char_seq(contents, char_maxlen=cfg.char_seq_maxlen, mode='post', keep=False, verbost=False):
    unknow_index = len(char_vocab)
    char_r = []
    for content in contents:
        char_c = []
        content = content.lower().strip()
        content = content.replace(" ", "")
        chars_line = " ".join(content)
        chars = chars_line.split(" ")
        for char in chars:
            if convert_num(char):
                char = 'NUM'
            if char in char_vocab:
                index = char_vocab[char]
            else:
                index = unknow_index
            char_c.append(index)
        char_c = np.array(char_c)
        char_r.append(char_c)
    char_seq = sequence.pad_sequences(char_r, maxlen=char_maxlen, padding=mode, truncating=mode, value=0)
    return char_seq


# In[24]:


def to_categorical(labels):
    y = []
    for label in labels:
        y_line = [0, 0, 0, 0]
        assert label < 4, 'label is %s' % label
        y_line[label] = 1
        y.append(y_line)
    y = np.array(y)
    return y


# In[22]:


# 得到一个tuple组成的list
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


# In[25]:


def batch_generator(contents, labels, batch_size=128, shuffle=True, keep=False, preprocessfunc=None):

    assert preprocessfunc != None
    sample_size = contents.shape[0]
    index_array = np.arange(sample_size)

    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start: batch_end]
            batch_contents = contents[batch_ids]
            batch_contents = preprocessfunc(batch_contents, keep=keep)
            batch_labels = to_categorical(labels[batch_ids])
            yield (batch_contents, batch_labels)


# In[26]:


def word_cnn_preprocess(contents, word_maxlen=cfg.word_seq_maxlen, keep=False):
    word_seq = get_word_seq(contents, word_maxlen=word_maxlen, keep=keep)
    return word_seq


# In[27]:


def word_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label,batch_size=batch_size, keep=keep, preprocessfunc=word_cnn_preprocess)

