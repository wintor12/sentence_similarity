from nltk.corpus import reuters 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from time import time


n_topics = 30
categories = ["acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade"]
docs = reuters.fileids(categories);
## Only keep documents that have one category
docs = list(filter(lambda doc: len(reuters.categories(doc)) == 1, docs))

train = list(filter(lambda doc: doc.startswith("train"), docs));
test = list(filter(lambda doc: doc.startswith("test"), docs));

train_docs = list(reuters.raw(doc) for doc in train)
test_docs = list(reuters.raw(doc) for doc in test)

train_target = np.asarray(list(reuters.categories(doc) for doc in train)).ravel()
test_target = np.asarray(list(reuters.categories(doc) for doc in test)).ravel()

vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 2, stop_words = 'english')
## Using tfidf

t0 = time()
vectors = vectorizer.fit_transform(train_docs)
vectors_test = vectorizer.transform(test_docs)
print vectors.shape

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(vectors, train_target)
pred = neigh.predict(vectors_test)
# # print float(list(test.target[:100]==r).count(True))/len(r)
print "tf-idf result: "
print metrics.accuracy_score(test_target,pred)
print("done in %0.3fs" % (time() - t0))

## Using LSA in tfidf
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
t0 = time()
vectors = vectorizer.fit_transform(train_docs)
vectors_test = vectorizer.transform(test_docs)
svd = TruncatedSVD(n_topics)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(vectors)
print X.shape
X_test = lsa.transform(vectors_test) ##can not use fit-transform here
print X_test.shape

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, train_target)
pred = neigh.predict(X_test)
print "LSA result: "
print metrics.accuracy_score(test_target,pred)
print("done in %0.3fs" % (time() - t0))


## Using NMF
from sklearn.decomposition import NMF
t0 = time()
vectors = vectorizer.fit_transform(train_docs)
vectors_test = vectorizer.transform(test_docs)

nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
X = nmf.fit_transform(vectors)
print X.shape
X_test = nmf.transform(vectors_test)
print X_test.shape

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, train_target)
pred = neigh.predict(X_test)
print "NMF result: "
print metrics.accuracy_score(test_target,pred)
print("done in %0.3fs" % (time() - t0))



## Using bag of words
from sklearn.feature_extraction.text import CountVectorizer
t0 = time()
vectorizer = CountVectorizer(max_df = 0.95, min_df = 2, stop_words = 'english')
vectors = vectorizer.fit_transform(train_docs)
vectors_test = vectorizer.transform(test_docs)
print vectors.shape

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(vectors, train_target)
pred = neigh.predict(vectors_test)
print "BOW result: "
print metrics.accuracy_score(test_target,pred)
print("done in %0.3fs" % (time() - t0))

## Using LDA in bow
from sklearn.decomposition import LatentDirichletAllocation
t0 = time()
vectors = vectorizer.fit_transform(train_docs)
vectors_test = vectorizer.transform(test_docs)

lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
X = lda.fit_transform(vectors)
print X.shape
X_test = lda.transform(vectors_test)
print X_test.shape

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, train_target)
pred = neigh.predict(X_test)
print "LDA result: "
print metrics.accuracy_score(test_target,pred)
print("done in %0.3fs" % (time() - t0))


## Using word2vec + tfidf
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import euclidean_distances
wv = Word2Vec.load_word2vec_format("/home/tong/Documents/python/GoogleNews-vectors-negative300.bin.gz", binary = True)

t0 = time()
vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 2, stop_words = 'english')
vectors = vectorizer.fit_transform(train_docs)
vectors_test = vectorizer.transform(test_docs)

vs = np.zeros((vectors.shape[1],300))
for word in vectorizer.vocabulary_.keys():
	if word in wv:
		vs[vectorizer.vocabulary_[word]] = wv[word]

v = np.zeros((vectors.shape[0], 300))
vect = vectors.todense()
v = np.dot(vect,vs)
vect_test = vectors_test.todense()
v_test = np.dot(vect_test, vs)
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(v, train_target)
pred = neigh.predict(v_test)
print metrics.accuracy_score(test_target,pred)
print("done in %0.3fs" % (time() - t0))


## Using glove + tfidf

import sys, codecs

vector_file = "/home/tong/Documents/python/glove.42B.300d.txt"
n_words = 1000000

numpy_arrays = []
labels_array = []
with codecs.open(vector_file, 'r', 'utf-8') as f:
	for c, r in enumerate(f):
		sr = r.split()
		labels_array.append(sr[0])
		numpy_arrays.append(np.array([float(i) for i in sr[1:]]) )
		if c == n_words:
			break

t0 = time()
vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 5, stop_words = 'english')
vectors = vectorizer.fit_transform(train_docs)
vectors_test = vectorizer.transform(test_docs)

vs = np.zeros((vectors.shape[1],300))
for word in vectorizer.vocabulary_.keys():
	if word in labels_array:
		vs[vectorizer.vocabulary_[word]] = numpy_arrays[labels_array.index(word)]

v = np.zeros((vectors.shape[0], 300))
vect = vectors.todense()
v = np.dot(vect,vs)
vect_test = vectors_test.todense()
v_test = np.dot(vect_test, vs)
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(v, train_target)
pred = neigh.predict(v_test)
print metrics.accuracy_score(test_target,pred)
print("done in %0.3fs" % (time() - t0))

