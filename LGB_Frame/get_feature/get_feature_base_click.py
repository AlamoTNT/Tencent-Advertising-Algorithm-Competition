# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:51:07 2020

@author: Alamo
"""

import pandas as pd


path = '../data/'

"""
click_log.csv中的一些基础统计特征
"""
train_click_df = pd.read_csv(path+'train_preliminary/click_log.csv')
test_click_df = pd.read_csv(path+'test/click_log.csv')

whole_click_df = pd.concat([train_click_df, test_click_df]).reset_index()
del whole_click_df['creative_id']

whole_click_df = whole_click_df.sort_values(by=['user_id', 'time'], ascending=[True, True])

data_feature = pd.DataFrame()
ALL_DAY_COUNT = len(whole_click_df['time'].unique())

data_feature['user_id'] = whole_click_df['user_id'].unique()
data_feature.set_index(keys=['user_id'], inplace=True)

data_feature['click_day_count'] = whole_click_df['user_id'].value_counts().sort_index()
data_feature['click_day_rate'] = data_feature['click_day_count'].apply(lambda x: x/ALL_DAY_COUNT)
data_feature['click_day_range'] = whole_click_df.groupby(['user_id'])['time'].agg({list}).reset_index()['list'].apply(lambda x: max(x)-min(x))


data_feature['click_times_mean'] = whole_click_df.groupby(['user_id'])['click_times'].agg('mean')
data_feature['click_times_std'] = whole_click_df.groupby(['user_id'])['click_times'].agg('std')
data_feature['click_times_max'] = whole_click_df.groupby(['user_id'])['click_times'].agg('max')
data_feature['click_times_min'] = whole_click_df.groupby(['user_id'])['click_times'].agg('min')

data_feature.to_csv(path+'feature/click_base_feature.csv')





































