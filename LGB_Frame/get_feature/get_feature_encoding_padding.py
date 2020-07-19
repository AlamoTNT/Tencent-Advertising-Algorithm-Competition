# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:15:27 2020

@author: Alamo
"""


import numpy as np
import pandas as pd
import scipy.sparse
from gensim import corpora, models 
from gensim.models import Word2Vec  

import warnings
warnings.filterwarnings("ignore")


path = '../data/'

"""
train_click_df = pd.read_csv(path+'train_preliminary/click_log.csv')
test_click_df = pd.read_csv(path+'test/click_log.csv')

whole_click_df = pd.concat([train_click_df, test_click_df])

def count_tf(x):
    return ' '.join([''+str(x['creative_id']) for _ in range(x['click_times'])])

whole_click_df['creative_id_tf'] = whole_click_df.apply(lambda x: count_tf(x), axis=1)


from tqdm import tqdm, tqdm_pandas
tqdm_pandas(tqdm())
def dealed_row(row):
    creative_id_list = list(row['creative_id_tf'])
    return ' '.join(creative_id_list)

data_feature = whole_click_df.groupby('user_id').progress_apply(lambda row: dealed_row(row)).reset_index()
#data_feature = pd.merge(whole_click_df, data_feature, on='user_id', how='left')
data_feature.to_csv(path+'temp_data_feature.csv')
"""
#上面的过程比较慢, 直接保存了方便读取
data_feature = pd.read_csv(path+'temp_data_feature.csv', index_col=[0])
corpus = data_feature['0'].tolist()
del data_feature

#对语料进行分词
#sentence_list = []
#for i in range(len(corpus)):
#    sentence_list.append(corpus[i].split(' '))

all_text = ' '.join(corpus)
words = all_text.split()

#Encoding the words
from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)

#构建字典保存函数
import json
import datetime
class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):                                 
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)
        
vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}
np.save(path+'vocab_to_int.npy', vocab_to_int)


## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
user_ints = []
for sentence in corpus:
    user_ints.append([vocab_to_int[word] for word in sentence.split()])

#Padding sequences
#选择每个user点击广告序列的长度

def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
 
    # for each review, I grab that review and 
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features


click_len = 200
 
features = pad_features(user_ints, seq_length=click_len)
 
## test statements - do not change - ##
assert len(features)==len(user_ints), "Your features should have as many rows as reviews."
assert len(features[0])==click_len, "Each feature row should contain seq_length values."


feature_df = pd.DataFrame(features)
feature_df['user_id'] = data_feature['user_id']

feature_df.to_csv(path+'feature_lstm.csv')




































"""
# ===============word2vec词向量+tfidf==================  
def sentenceByW2VTfidf(corpus_tfidf, token2id, sentenceList, model, embeddingSize):  
    sentenceSet = []  
    for i in range(len(sentenceList)):  
        # 将所有词向量的woed2vec向量相加到句向量  
        sentenceVector = np.zeros(embeddingSize)  
        # 计算每个词向量的权重，并将词向量加到句向量  
        sentence = sentenceList[i]  
        sentence_tfidf = corpus_tfidf[i]  
        dict_tfidf = list_dict(sentence_tfidf)  
        for word in sentence:  
            if word in token2id.keys():
                tifidf_weight = dict_tfidf.get(str(token2id[word]))  
            else:
                tifidf_weight = 0
            sentenceVector = np.add(sentenceVector, tifidf_weight * model[word])  
        sentenceVector = np.divide(sentenceVector, len(sentence))  
        # 存储句向量  
        sentenceSet.append(sentenceVector)  
    return sentenceSet  

def list_dict(list_data):  
    list_data=list(map(lambda x:{str(x[0]):x[1]},list_data))  
    dict_data = {}  
    for i in list_data:  
        key, = i  
        value, = i.values()  
        dict_data[key] = value  
    return dict_data  

dictionary = corpora.Dictionary(sentence_list)
print('文档数目: ', dictionary.num_docs)    #用户的数目
print('所有词的个数: ', dictionary.num_pos) #所有被点击的次数

#dictionary.save('./tfidf/ths_dict.dict')  # 保存生成的词典
#dictionary = corpora.Dictionary.load('./tfidf/ths_dict.dict')  # 加载

token2id = dictionary.token2id
corpus = [dictionary.doc2bow(text) for text in sentence_list]
#corpora.MmCorpus.serialize('./tfidf/ths_corpuse.mm', corpus)  # 将生成的语料保存成MM文件
#corpus = corpora.MmCorpus('./tfidf/ths_corpuse.mm')  # 加载
print('dictionary prepared!')  

tfidf = models.TfidfModel(corpus=corpus, dictionary=dictionary)
wdfs = tfidf.dfs
corpus_tfidf = tfidf[corpus]

model = Word2Vec(sentence_list, size=128, window=5, min_count=1, workers=12, sg=1, iter=10, seed=1017)
# 词向量tfidf加权得到句向量  
sentence_vecs = sentenceByW2VTfidf(corpus_tfidf, token2id, sentence_list, model, 128)

tfidf_w2v_vec = pd.DataFrame(sentence_vecs)
tfidf_w2v_vec.columns = ['tfidf_w2v_vec_{}'.format(i) for i in range(1, 129)]
tfidf_w2v_vec['user_id'] = data_feature['user_id']
tfidf_w2v_vec.set_index(keys=['user_id'], inplace=True)
tfidf_w2v_vec.to_csv(path+'feature/tfidf_w2v_vec_feature.csv')
"""






