# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:19:22 2020

@author: Alamo
"""


import pandas as pd
import numpy as np

# 减小内存消耗
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

feature_path = '../data/feature/'

#w2v_feature_creative_id = pd.read_csv(feature_path+'creativeid_w2v_avg_feature.csv')
#w2v_feature_ad_id = pd.read_csv(feature_path+'adid_w2v_avg_feature.csv')
#del w2v_feature_ad_id['user_id']
#tfidf_w2v_vec_feature = pd.read_csv(feature_path+'tfidf_w2v_vec_feature.csv')
#w2v_feature_creative_id['user_id']
base_feature_click = pd.read_csv(feature_path+'click_base_feature.csv')
#del base_feature_click['user_id']

creative_id_tfidf_bnb_feature = pd.read_csv(feature_path+'creative_id_tfidf_bnb_error_single_classfiy.csv')
creative_id_tfidf_lr_feature = pd.read_csv(feature_path+'creative_id_tfidf_lr_error_single_classfiy.csv')
creative_id_tfidf_lsvc_feature = pd.read_csv(feature_path+'creative_id_tfidf_lsvc_error_single_classfiy.csv')
creative_id_tfidf_mnb_feature = pd.read_csv(feature_path+'creative_id_tfidf_mnb_error_single_classfiy.csv')
creative_id_tfidf_pac_feature = pd.read_csv(feature_path+'creative_id_tfidf_pac_error_single_classfiy.csv')
creative_id_tfidf_ridge_feature = pd.read_csv(feature_path+'creative_id_tfidf_ridge_error_single_classfiy.csv')
creative_id_tfidf_sgd_feature = pd.read_csv(feature_path+'creative_id_tfidf_sgd_error_single_classfiy.csv')

"""
ad_id_tfidf_bnb_feature = pd.read_csv(feature_path+'ad_id_tfidf_bnb_error_single_classfiy.csv')
ad_id_tfidf_lr_feature = pd.read_csv(feature_path+'ad_id_tfidf_lr_error_single_classfiy.csv')
ad_id_tfidf_lsvc_feature = pd.read_csv(feature_path+'ad_id_tfidf_lsvc_error_single_classfiy.csv')
ad_id_tfidf_mnb_feature = pd.read_csv(feature_path+'ad_id_tfidf_mnb_error_single_classfiy.csv')
ad_id_tfidf_pac_feature = pd.read_csv(feature_path+'ad_id_tfidf_pac_error_single_classfiy.csv')
ad_id_tfidf_ridge_feature = pd.read_csv(feature_path+'ad_id_tfidf_ridge_error_single_classfiy.csv')
ad_id_tfidf_sgd_feature = pd.read_csv(feature_path+'ad_id_tfidf_sgd_error_single_classfiy.csv')

advertiser_id_tfidf_bnb_feature = pd.read_csv(feature_path+'advertiser_id_tfidf_bnb_error_single_classfiy.csv')
advertiser_id_tfidf_lr_feature = pd.read_csv(feature_path+'advertiser_id_tfidf_lr_error_single_classfiy.csv')
advertiser_id_tfidf_lsvc_feature = pd.read_csv(feature_path+'advertiser_id_tfidf_lsvc_error_single_classfiy.csv')
advertiser_id_tfidf_mnb_feature = pd.read_csv(feature_path+'advertiser_id_tfidf_mnb_error_single_classfiy.csv')
advertiser_id_tfidf_pac_feature = pd.read_csv(feature_path+'advertiser_id_tfidf_pac_error_single_classfiy.csv')
advertiser_id_tfidf_ridge_feature = pd.read_csv(feature_path+'advertiser_id_tfidf_ridge_error_single_classfiy.csv')
advertiser_id_tfidf_sgd_feature = pd.read_csv(feature_path+'advertiser_id_tfidf_sgd_error_single_classfiy.csv')

product_id_tfidf_bnb_feature = pd.read_csv(feature_path+'product_id_tfidf_bnb_error_single_classfiy.csv')
product_id_tfidf_lr_feature = pd.read_csv(feature_path+'product_id_tfidf_lr_error_single_classfiy.csv')
product_id_tfidf_lsvc_feature = pd.read_csv(feature_path+'product_id_tfidf_lsvc_error_single_classfiy.csv')
product_id_tfidf_mnb_feature = pd.read_csv(feature_path+'product_id_tfidf_mnb_error_single_classfiy.csv')
product_id_tfidf_pac_feature = pd.read_csv(feature_path+'product_id_tfidf_pac_error_single_classfiy.csv')
product_id_tfidf_ridge_feature = pd.read_csv(feature_path+'product_id_tfidf_ridge_error_single_classfiy.csv')
product_id_tfidf_sgd_feature = pd.read_csv(feature_path+'product_id_tfidf_sgd_error_single_classfiy.csv')

industry_tfidf_bnb_feature = pd.read_csv(feature_path+'industry_tfidf_bnb_error_single_classfiy.csv')
industry_tfidf_lr_feature = pd.read_csv(feature_path+'industry_tfidf_lr_error_single_classfiy.csv')
industry_tfidf_lsvc_feature = pd.read_csv(feature_path+'industry_tfidf_lsvc_error_single_classfiy.csv')
industry_tfidf_mnb_feature = pd.read_csv(feature_path+'industry_tfidf_mnb_error_single_classfiy.csv')
industry_tfidf_pac_feature = pd.read_csv(feature_path+'industry_tfidf_pac_error_single_classfiy.csv')
industry_tfidf_ridge_feature = pd.read_csv(feature_path+'industry_tfidf_ridge_error_single_classfiy.csv')
industry_tfidf_sgd_feature = pd.read_csv(feature_path+'industry_tfidf_sgd_error_single_classfiy.csv')
"""
#frames = [w2v_feature_creative_id, base_feature_click]
frames = [base_feature_click, 
          creative_id_tfidf_bnb_feature, creative_id_tfidf_lr_feature, creative_id_tfidf_lsvc_feature, creative_id_tfidf_mnb_feature,
          creative_id_tfidf_pac_feature, creative_id_tfidf_ridge_feature, creative_id_tfidf_sgd_feature
          ]
          
"""
ad_id_tfidf_bnb_feature, ad_id_tfidf_lr_feature, ad_id_tfidf_lsvc_feature, ad_id_tfidf_mnb_feature,
ad_id_tfidf_pac_feature, ad_id_tfidf_ridge_feature, ad_id_tfidf_sgd_feature,
advertiser_id_tfidf_bnb_feature, advertiser_id_tfidf_lr_feature, advertiser_id_tfidf_lsvc_feature, advertiser_id_tfidf_mnb_feature,
advertiser_id_tfidf_pac_feature, advertiser_id_tfidf_ridge_feature, advertiser_id_tfidf_sgd_feature,
product_id_tfidf_bnb_feature, product_id_tfidf_lr_feature, product_id_tfidf_lsvc_feature, product_id_tfidf_mnb_feature,
product_id_tfidf_pac_feature, product_id_tfidf_ridge_feature, product_id_tfidf_sgd_feature,
industry_tfidf_bnb_feature, industry_tfidf_lr_feature, industry_tfidf_lsvc_feature, industry_tfidf_mnb_feature,
industry_tfidf_pac_feature, industry_tfidf_ridge_feature, industry_tfidf_sgd_feature]
"""
all_feature = pd.concat(frames, axis=1, join='inner')

#all_feature = pd.read_csv(feature_path+'all_feature.csv', index_col=[0])
all_feature = reduce_mem_usage(all_feature)
all_feature.to_csv(feature_path+'all_feature.csv')
