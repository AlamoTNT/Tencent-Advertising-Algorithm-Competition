# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:25:18 2020

@author: Alamo
"""


import pandas as pd
from sklearn import preprocessing

path = '../data/'

train_data = pd.read_csv(path+'train_preliminary/user.csv')
all_feature = pd.read_csv(path+'feature/all_feature.csv', index_col=[0])

train_data['label'] = train_data['gender'].astype(str) + '-' + train_data['age'].astype(str)
label_encoder = preprocessing.LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
data_all = pd.merge(left=all_feature, right=train_data, on='user_id', how='left')

train = data_all[:len(train_data)]
test = data_all[len(train_data):]

#rnn_sub = pd.read_csv('rnn_sub.csv', index_col=[0])
cnn_sub = pd.read_csv('../model_props/test_props/cnn_test_props_advertiser_id.csv', index_col=[0])

#sub = (cnn_sub + rnn_sub)/2

sub = pd.DataFrame(cnn_sub)
cols = [x for x in range(20)]
cols = label_encoder.inverse_transform(cols)
sub.columns = cols

submission = pd.DataFrame()
submission['user_id'] = test['user_id'].values

def get_age(row):
    argidx = row[row==row.max()].index
    return int(argidx[0][-1] if len(argidx[0])==3 else argidx[0][-2:])

def get_gender(row):
    argidx = row[row==row.max()].index
    return int(argidx[0][0])
    
submission['predicted_age'] = sub.apply(lambda row: get_age(row), axis=1)
submission['predicted_gender'] = sub.apply(lambda row: get_gender(row), axis=1)


submission.to_csv(path+'submission.csv', encoding='utf-8-sig')