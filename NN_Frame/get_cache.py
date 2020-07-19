# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 07:17:47 2020

@author: Alamo
"""

import torch
from utils import read_corpus
from sklearn.model_selection import train_test_split


ad_data, labels = read_corpus('./corpus/texts_ad.csv')
ad_te_data = ad_data[900000:]
del labels, ad_data

advertiser_data, labels = read_corpus('./corpus/texts_advertiser.csv')
advertiser_te_data = advertiser_data[900000:]
del labels, advertiser_data

creative_data, labels = read_corpus('./corpus/texts_creative.csv')
creative_te_data = creative_data[900000:]
del labels, creative_data

industry_data, labels = read_corpus('./corpus/texts_industry.csv')
industry_te_data = industry_data[900000:]
del labels, industry_data

product_data, labels = read_corpus('./corpus/texts_product.csv')
product_te_data = product_data[900000:]
del labels, product_data

test_data = [(ad, advertiser, creative, industry, product) for ad, advertiser, creative, industry, product in zip(ad_te_data, advertiser_te_data, creative_te_data, industry_te_data, product_te_data)]

torch.save(test_data,'./cache/cache_test_data')

del ad_te_data, advertiser_te_data, creative_te_data, industry_te_data, product_te_data

X_train, X_val, y_train, y_val= train_test_split(train_data, labels, test_size=0.2, random_state=1017)


train_data = [(text,labs) for text,labs in zip(X_train, y_train)]
dev_data = [(text,labs) for text,labs in zip(X_val, y_val)]
torch.save(train_data,'./cache/cache_train_data')
torch.save(dev_data,'./cache/cache_dev_data')





