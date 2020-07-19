import math
import random
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm, tqdm_pandas, trange
import warnings

warnings.filterwarnings("ignore")
tqdm_pandas(tqdm())

def read_corpus(file_path):
    """读取语料
    :param file_path:
    :param type:
    :return:
    """

    texts = pd.read_csv(file_path+'texts_ad.csv', index_col=[0])
    texts_list = texts[0].tolist()
    
    corpus = []
    for i in trange(len(texts_list)):
        corpus.append(texts_list[i].split(' '))

    
    user_info = pd.read_csv(file_path+'user.csv')
    
    user_info['label'] = user_info['gender'].astype(str) + '-' + user_info['age'].astype(str)
    label_encoder = preprocessing.LabelEncoder()
    user_info['label'] = label_encoder.fit_transform(user_info['label'])
    del user_info['age'], user_info['gender']
    labels = user_info['label'].tolist()
    
    return (corpus, labels)

def pad_sents(sents, pad_token):
    """pad句子"""
    sents_padded = []
    #lengths = [len(s) for s in sents]
    #max_len = max(lengths)
    max_len = 300
    for sent in sents:
        if len(sent)>=max_len:
            sents_padded.append(sent[:max_len])
        else:
            sent_padded = sent + [pad_token] * (max_len - len(sent))
            sents_padded.append(sent_padded)
    return sents_padded

def batch_iter(data, batch_size, test_batch=False, shuffle=False):
    """
    batch数据
    :param data: list of tuple
    :param batch_size:
    :param shuffle:
    :return:
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    if shuffle:
        random.shuffle(index_array)
    
    if not test_batch:
        for i in range(batch_num):
            indices = index_array[i*batch_size:(i+1)*batch_size]
            examples = [data[idx] for idx in indices]
            #examples = sorted(examples, key=lambda x: len(x[1]), reverse=True)
            src_sents = [e[0] for e in examples]
            labels = [e[1] for e in examples]
    
            yield src_sents, labels
    else:
        for i in range(batch_num):
            indices = index_array[i*batch_size:(i+1)*batch_size]
            src_sents = [data[idx] for idx in indices]
            
            yield  src_sents

def batch_iter_props(data, batch_size):
    """
    batch数据
    :param data: list of tuple
    :param batch_size:
    :param shuffle:
    :return:
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    for i in range(batch_num):
        indices = index_array[i*batch_size:(i+1)*batch_size]
        examples = [data[idx] for idx in indices]
        #examples = sorted(examples, key=lambda x: len(x[1]), reverse=True)
        src_sents = [e[0] for e in examples]
        #labels = [e[1] for e in examples]

        yield src_sents
    
        