import os

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_20newsgroups
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import euclidean_distances

def findClass(v, train):



wv = Word2Vec.load_word2vec_format("/home/tong/Documents/python/GoogleNews-vectors-negative300.bin.gz", binary = True) 

train = fetch_20newsgroups(shuffle=True, random_state=1, subset='train', remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(shuffle=True, random_state=1, subset='test', remove=('headers', 'footers', 'quotes'))

vect = CountVectorizer(max_df = 0.95, min_df = 5, stop_words = 'english').fit(train.data)

voc = [i for i,k in vect.vocabulary_.iteritems() if i in wv]
voc_vec = {i:wv[i] for i in voc}

from sklearn.metrics import euclidean_distances
W_ = [voc_vec[w] for w in voc]
# D_ = euclidean_distances(W_)
D_ = np.ones((len(voc),len(voc)))
for i in range(len(voc)):
	for j in range(i):
		D_[i][j] = wv.similarity(voc[i], voc[j])
		D_[j][i] = D_[i][j]