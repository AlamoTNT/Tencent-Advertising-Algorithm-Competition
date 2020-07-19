# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:45:32 2020

@author: Alamo
"""

import numpy as np
import pandas as pd
import scipy.sparse



path = './sparse_csr_matrix_save/'

whole_click_df = pd.read_csv(path+'whole_click_df.csv')

from tqdm import tqdm, tqdm_pandas
tqdm.pandas(tqdm())
def dealed_row(row):
    creative_id_list = list(row['creative_id_tf'])
    return ' '.join(creative_id_list)

data_feature = whole_click_df.groupby('user_id').progress_apply(lambda row:dealed_row(row)).reset_index()
#data_feature = pd.merge(whole_click_df, data_feature, on='user_id', how='left')
del data_feature['user_id']


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
count_vec = CountVectorizer(min_df=30)
count_csr_basic = count_vec.fit_transform(data_feature[0])

#save_path = '../data/feature/'
#scipy.sparse.save_npz(save_path+'count_csr_basic.npz', count_csr_basic)

#csr_matrix_variable = scipy.sparse.load_npz('path.npz')

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


path = '../data/'

train_data = pd.read_csv(path+'train_preliminary/user.csv')
all_feature = pd.read_csv(path+'feature/all_feature.csv', index_col=[0])

train_data['label'] = train_data['gender'].astype(str) + '-' + train_data['age'].astype(str)
label_encoder = preprocessing.LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
data_all = pd.merge(left=all_feature, right=train_data, on='user_id', how='left')

train = data_all[:len(train_data)]
test = data_all[len(train_data):]
del data_all
gc.collect()

use_feature = all_feature.columns[1:]
use_feature

X_train = train[use_feature]
X_test = test[use_feature]
Y = train['label']
#X_train, X_test, Y = X_train.values, X_test.values, Y.values

X_train = scipy.sparse.hstack((scipy.sparse.csr_matrix(train[use_feature]), count_csr_basic[:900000])).tocsr()
X_test = scipy.sparse.hstack((scipy.sparse.csr_matrix(test[use_feature]), count_csr_basic[900000:])).tocsr()
Y = Y.values

kfold = StratifiedKFold(n_splits=2, random_state=1017, shuffle=True)
sub = np.zeros((X_test.shape[0], 20))
for i, (train_index, test_index) in enumerate(kfold.split(X_train, Y)):
    print('{}-th fold: '.format(i+1))
    #X_tr, X_vl, y_tr, y_vl = X_train.iloc[train_index], X_train.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]
    X_tr, X_vl, y_tr, y_vl = X_train[train_index], X_train[test_index], Y[train_index], Y[test_index]
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=[-1])
    dvalid = lgb.Dataset(X_vl, y_vl, reference=dtrain)
    params = {
        'boosting_type': 'gbdt',
        'max_depth':6,
        'max_bin':63,
        'metric': {'multi_logloss'},
        'num_class':20,
        'objective':'multiclass',
        'num_leaves':7,
        'subsample': 0.9,
        'colsample_bytree': 0.2,
        'lambda_l1':0.0001,
        'lambda_l2':0.00111,
        'subsample_freq':12,
        'learning_rate': 0.03,
        'min_child_weight':12,
        'device':'gpu',
        'gpu_platform_id':0,
        'gpu_device_id':0,
        'gpu_use_dp':'false'
    }

    model = lgb.train(params,
                      train_set=dtrain,
                      num_boost_round=6000,      #可以继续增大
                      valid_sets=dvalid,
                      early_stopping_rounds=100,
                      verbose_eval=100)

    sub += model.predict(X_test, num_iteration=model.best_iteration)/kfold.n_splits


sub = pd.DataFrame(sub)
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





