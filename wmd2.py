import os

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_20newsgroups
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from pyemd import emd

wv = Word2Vec.load_word2vec_format("/home/tong/Documents/python/GoogleNews-vectors-negative300.bin.gz", binary = True)

newsgroups = fetch_20newsgroups()
docs, y = newsgroups.data, newsgroups.target
docs_train, docs_test, y_train, y_test = train_test_split(docs, y,train_size=100,test_size=300,random_state=0)

vectorizer = CountVectorizer(max_df = 0.95, min_df = 3, stop_words = 'english')
vect = vectorizer.fit(docs_train + docs_test)
common = [word for word in vect.get_feature_names() if word in wv]
W_common = [wv[w] for w in common]
vect = CountVectorizer(vocabulary=common, dtype=np.double)
X_train = vect.fit_transform(docs_train)
X_test = vect.transform(docs_test)

D_ = euclidean_distances(W_common)
D_ = D_.astype(np.double)
D_ /= D_.max()

def mydist(x, y):
	x = x.astype(np.double)
	y = y.astype(np.double)
	x /= x.sum()
	y /= y.sum()
	return emd(x, y, D_)

X_train_dense = X_train.todense()
X_test_dense = X_test.todense()
neigh = KNeighborsClassifier(n_neighbors=5, metric = 'pyfunc', func=mydist)
neigh.fit(X_train_dense, y_train)







