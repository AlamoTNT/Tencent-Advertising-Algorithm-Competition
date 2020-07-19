# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:52:28 2020

@author: Alamo
"""

import numpy as np
import pandas as pd
import scipy.sparse

import warnings
warnings.filterwarnings("ignore")


path = '../data/'

data_feature = pd.read_csv('../model_/corpus/texts_advertiser.csv', index_col=[0])
del data_feature['user_id']


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
count_vec = CountVectorizer(min_df=30)
count_csr_basic = count_vec.fit_transform(data_feature['0'])
tfidf_vec = TfidfVectorizer(min_df=30)
tfidf_vec_basic = tfidf_vec.fit_transform(data_feature['0'])

data_feature = scipy.sparse.csr_matrix(scipy.sparse.hstack([count_csr_basic, tfidf_vec_basic]))
del count_csr_basic, tfidf_vec_basic

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 1017

user_df = pd.read_csv(path+'train_preliminary/user.csv')

def get_label(row):
    if row['gender'] == 1:
        return row['age']
    else:
        return row['age'] + 10

user_df['label'] = user_df.apply(lambda row:get_label(row), axis=1)

train_feature = data_feature[:len(user_df)]
score = user_df['label']
test_feature = data_feature[len(user_df):]
number = len(np.unique(score))

del user_df

# 五则交叉验证
n_folds = 5
print('处理完毕')
kfold = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)

########################### lr(LogisticRegression) ################################
print('lr stacking')
stack_train = np.zeros((train_feature.shape[0], number))
stack_test = np.zeros((test_feature.shape[0], number))
score_va = 0

for i, (tr, va) in enumerate(kfold.split(train_feature, score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    clf = LogisticRegression(random_state=RANDOM_SEED, C=8)
    clf.fit(train_feature[tr], score[tr])
    score_va = clf.predict_proba(train_feature[va])
    score_te = clf.predict_proba(test_feature)
    print('得分' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['advertiser_id_tfidf_lr_classfiy_{}'.format(i)] = np.around(stack[:, i], 6) #6是decimals参数, 指保留6位小数
df_stack.to_csv(path+'feature/advertiser_id_tfidf_lr_error_single_classfiy.csv', index=None, encoding='utf8')
print('lr特征已保存\n')

########################### SGD(随机梯度下降) ################################
print('sgd stacking')
stack_train = np.zeros((train_feature.shape[0], number))
stack_test = np.zeros((test_feature.shape[0], number))
score_va = 0

for i, (tr, va) in enumerate(kfold.split(train_feature, score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    sgd = SGDClassifier(random_state=RANDOM_SEED, loss='log')
    sgd.fit(train_feature[tr], score[tr])
    score_va = sgd.predict_proba(train_feature[va])
    score_te = sgd.predict_proba(test_feature)
    print('得分' + str(mean_squared_error(score[va], sgd.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['advertiser_id_tfidf_sgd_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv(path+'feature/advertiser_id_tfidf_sgd_error_single_classfiy.csv', index=None, encoding='utf8')
print('sgd特征已保存\n')

########################### pac(PassiveAggressiveClassifier) ################################
print('PAC stacking')
stack_train = np.zeros((train_feature.shape[0], number))
stack_test = np.zeros((test_feature.shape[0], number))
score_va = 0

for i, (tr, va) in enumerate(kfold.split(train_feature, score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    pac = PassiveAggressiveClassifier(random_state=RANDOM_SEED)
    pac.fit(train_feature[tr], score[tr])
    score_va = pac._predict_proba_lr(train_feature[va])
    score_te = pac._predict_proba_lr(test_feature)
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], pac.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['advertiser_id_tfidf_pac_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv(path+'feature/advertiser_id_tfidf_pac_error_single_classfiy.csv', index=None, encoding='utf8')
print('pac特征已保存\n')


########################### ridge(RidgeClassfiy) ################################
print('RidgeClassfiy stacking')
stack_train = np.zeros((train_feature.shape[0], number))
stack_test = np.zeros((test_feature.shape[0], number))
score_va = 0

for i, (tr, va) in enumerate(kfold.split(train_feature, score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    ridge = RidgeClassifier(random_state=RANDOM_SEED)
    ridge.fit(train_feature[tr], score[tr])
    score_va = ridge._predict_proba_lr(train_feature[va])
    score_te = ridge._predict_proba_lr(test_feature)
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], ridge.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['advertiser_id_tfidf_ridge_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv(path+'feature/advertiser_id_tfidf_ridge_error_single_classfiy.csv', index=None, encoding='utf8')
print('ridge特征已保存\n')


########################### bnb(BernoulliNB) ################################
print('BernoulliNB stacking')
stack_train = np.zeros((train_feature.shape[0], number))
stack_test = np.zeros((test_feature.shape[0], number))
score_va = 0

for i, (tr, va) in enumerate(kfold.split(train_feature, score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    bnb = BernoulliNB()
    bnb.fit(train_feature[tr], score[tr])
    score_va = bnb.predict_proba(train_feature[va])
    score_te = bnb.predict_proba(test_feature)
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], bnb.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['advertiser_id_tfidf_bnb_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv(path+'feature/advertiser_id_tfidf_bnb_error_single_classfiy.csv', index=None, encoding='utf8')
print('BernoulliNB特征已保存\n')

########################### mnb(MultinomialNB) ################################
print('MultinomialNB stacking')
stack_train = np.zeros((train_feature.shape[0], number))
stack_test = np.zeros((test_feature.shape[0], number))
score_va = 0

for i, (tr, va) in enumerate(kfold.split(train_feature, score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    mnb = MultinomialNB()
    mnb.fit(train_feature[tr], score[tr])
    score_va = mnb.predict_proba(train_feature[va])
    score_te = mnb.predict_proba(test_feature)
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], mnb.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['advertiser_id_tfidf_mnb_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv(path+'feature/advertiser_id_tfidf_mnb_error_single_classfiy.csv', index=None, encoding='utf8')
print('MultinomialNB特征已保存\n')

############################ Linersvc(LinerSVC) ################################
print('LinerSVC stacking')
stack_train = np.zeros((train_feature.shape[0], number))
stack_test = np.zeros((test_feature.shape[0], number))
score_va = 0

for i, (tr, va) in enumerate(kfold.split(train_feature, score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    lsvc = LinearSVC(random_state=RANDOM_SEED)
    lsvc.fit(train_feature[tr], score[tr])
    score_va = lsvc._predict_proba_lr(train_feature[va])
    score_te = lsvc._predict_proba_lr(test_feature)
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], lsvc.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
df_stack = pd.DataFrame()
for i in range(stack.shape[1]):
    df_stack['advertiser_id_tfidf_lsvc_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
df_stack.to_csv(path+'feature/advertiser_id_tfidf_lsvc_error_single_classfiy.csv', index=None, encoding='utf8')
print('LSVC特征已保存\n')


"""
kmeans_result = pd.DataFrame()
###### kmeans ###
def get_cluster(num_clusters):
    print('开始' + str(num_clusters))
    name = 'kmean'
    print(name)
    model = KMeans(n_clusters=num_clusters, max_iter=300, n_init=1, \
                        init='k-means++', n_jobs=10, random_state=RANDOM_SEED)
    result = model.fit_predict(data_feature)
    kmeans_result[name + 'word_' + str(num_clusters)] = result

get_cluster(5)
get_cluster(10)
get_cluster(19)
get_cluster(30)
get_cluster(40)
get_cluster(50)
get_cluster(60)
get_cluster(70)
kmeans_result.to_csv(path+'feature/creative_id_cluster_tfidf_feature.csv', index=False)
"""






















