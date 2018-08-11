
# coding: utf-8

# In[1]:


import os
import sys
BASE = '/home/wb/smp2018'
sys.path.append(BASE)


# In[2]:


import json
import pandas as pd
from init.config import Config
from pyltp import SentenceSplitter
from pyltp import Segmentor
from sklearn.model_selection import train_test_split


# In[3]:


cfg = Config()


# In[4]:


label_dict = {"自动摘要": 0, "机器翻译": 1, "机器作者": 2, "人类作者": 3}


# In[5]:


LTP_DATA_DIR = '/home/wb/ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model


# In[6]:


def create_training_data(save_all_samples=False, save_split_sample=False):
    lines = []
    with open(cfg.train_raw_org_path) as f:
        lines = f.readlines()
    
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    data = []
    total = len(lines)
    one = total // 100
    for i, l in enumerate(lines):
        js = json.loads(l)
        sen = js['内容'].replace('\r', '').replace('\t', '')
        sen = ' '.join(segmentor.segment(sen))
        js['内容'] = sen
        data.append(js)
        
        if i % one == 0:
            print('cut processing %s %%' % str((i * 100) // total))
    
    segmentor.release()  # 释放模型

    data = pd.DataFrame(data)

    data.rename(columns={'内容': 'content', '标签': 'label'}, inplace=True)

    data['label'] = data['label'].map(label_dict)

    print("total samples number:", len(data))
    
    if save_all_samples:
        data.fillna("", inplace=True)
        data.to_csv(cfg.train_all_data_path, index=False, sep='\t')
        
    train, val = train_test_split(data, test_size=0.1, shuffle=True, random_state=1)
    print("train samples number:", len(train))
    print("vali samples number:", len(val))
    if save_split_sample:
        train.to_csv(cfg.train_data_path, index=False, sep='\t')
        val.to_csv(cfg.test_data_path, index=False, sep='\t')
    return train, val


# - total samples number: 146341
# - train samples number: 131706
# - vali samples number: 14635

# In[7]:


def get_train_split_data():
    train = pd.read_csv(cfg.train_data_path, sep='\t')
    test = pd.read_csv(cfg.test_data_path, sep='\t')
    return train, test


# In[8]:


# train, val = create_training_data(True, True)


# In[9]:


def create_validation_data(save_samples=False):
    lines = []
    with open(cfg.val_raw_org_path) as f:
        lines = f.readlines()
        
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    data = []
    total = len(lines)
    one = total // 100
    for i, l in enumerate(lines):
        js = json.loads(l)
        sen = js['内容'].replace('\r', '').replace('\t', '')
        sen = ' '.join(segmentor.segment(sen))
        js['内容'] = sen
        data.append(js)
        
        if i % one == 0:
            print('cut processing %s %%' % str((i * 100) // total))
    
    segmentor.release()  # 释放模型
    
    data = pd.DataFrame(data)
    data.rename(columns={'内容': 'content'}, inplace=True)

    print("total samples number of validation:", len(data))
    
    if save_samples:
        data.fillna("", inplace=True)
        data.to_csv(cfg.vali_data_path, index=False, sep='\t')


# - total samples number of validation: 58537

# create_validation_data(True)

def get_train_all_data():
    train_all_data = pd.read_csv(cfg.train_all_data_path, sep='\t')
    return train_all_data


# get_train_all_data()



def get_validation_data():
    vali = pd.read_csv(cfg.vali_data_path, sep='\t')
    return vali

# df = get_validation_data()
# train, test = get_train_split_data()

