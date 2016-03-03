from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

train = fetch_20newsgroups(shuffle=True, random_state=1, subset='train')
test = fetch_20newsgroups(shuffle=True, random_state=1, subset='test')

vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 5, stop_words = 'english')
train_vecs = vectorizer.fit_transform(train.data)
train_vecs.shape
X = np.zeros(train_vecs.shape)
for i in range(len(train.data)):	
	X[i] = train_vecs[i].toarray().ravel()

vectorizer2 = TfidfVectorizer(vocabulary = vectorizer.vocabulary_)
test_vecs = vectorizer2.fit_transform(test.data)
test_vecs.shape

Z = np.zeros(test_vecs.shape)
for i in range(len(test.data)):	
	Z[i] = test_vecs[i].toarray().ravel()

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
neigh.fit(X, train.target)

r = neigh.predict(Z[:100])
print float(list(test.target[:100]==r).count(True))/len(r)
