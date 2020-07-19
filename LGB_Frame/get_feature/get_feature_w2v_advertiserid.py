# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:42:42 2020

@author: Alamo
"""

import pandas as pd
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec

path = '../data/'
train_ad_df = pd.read_csv(path+'train_preliminary/ad.csv', usecols=['creative_id', 'advertiser_id'])
test_ad_df = pd.read_csv(path+'test/ad.csv', usecols=['creative_id', 'ad_id'])
whole_ad_df = pd.concat([train_ad_df, test_ad_df])
whole_ad_df = whole_ad_df.drop_duplicates(subset=['creative_id'], keep='first')
del train_ad_df, test_ad_df

train_click_df = pd.read_csv(path+'train_preliminary/click_log.csv', usecols=['time', 'user_id', 'creative_id'])
test_click_df = pd.read_csv(path+'test/click_log.csv', usecols=['time', 'user_id', 'creative_id'])
whole_click_df = pd.concat([train_click_df, test_click_df])
#经分析, 并无重复记录
#whole_click_df = whole_click_df.drop_duplicates(subset=['time', 'user_id', 'creative_id'], keep='first')
del train_click_df, test_click_df, whole_click_df['time']

whole_click_ad_df = pd.merge(whole_click_df, whole_ad_df, how='left', on='creative_id')

#提取用户点击的广告序列, 并构成文本
doc = whole_click_ad_df.groupby(['user_id'])['advertiser_id'].agg({list}).reset_index()
document = doc['list'].values.tolist()

#转为字符串型才能进行训练
texts = [[str(word) for word in doc] for doc in document]

random_seed = 2020
w2v_model = Word2Vec(texts, size=128, window=5, min_count=1, workers=12, sg=1, iter=10, seed=random_seed)
w2v_model.wv.save_word2vec_format('./word2vec_model/advertiserid_w2v_128.txt')
del texts



def get_w2v_avg(doc, w2v_out_path, word2vec_Path):
    w2v_dim = 128

    model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_Path, binary=False)
    vacab = model.vocab.keys()

    w2v_feature = np.zeros((len(doc), w2v_dim))
    w2v_feature_avg = np.zeros((len(doc), w2v_dim))

    for i, line in enumerate(doc['list']):
        num = 0
        if line == '':
            w2v_feature_avg[i, :] = np.zeros(w2v_dim)
        else:
            for word in line:
                num += 1
                vec = model[str(word)] if str(word) in vacab else np.zeros(w2v_dim)
                w2v_feature[i, :] += vec
            w2v_feature_avg[i, :] = w2v_feature[i, :] / num
    w2v_avg = pd.DataFrame(w2v_feature_avg)
    w2v_avg.columns = ['w2v_avg_' + str(i) for i in range(1, w2v_dim+1)]
    w2v_avg['user_id'] = doc['user_id']
    w2v_avg.set_index(keys=['user_id'], inplace=True)
    w2v_avg.to_csv(w2v_out_path, encoding='utf-8', index=None)
    return w2v_avg


w2v_feat = get_w2v_avg(doc, path+'feature/adid_w2v_avg_feature.csv', './word2vec_model/advertiserid_w2v_128.txt')
