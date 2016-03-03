from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
# newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)
# newsgroups_test = fetch_20newsgroups(subset='test',categories = categories)
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 4, stop_words = 'english')
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)
print vectors.shape


neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(vectors, newsgroups_train.target)
pred = neigh.predict(vectors_test)
# # print float(list(test.target[:100]==r).count(True))/len(r)

print metrics.accuracy_score(newsgroups_test.target,pred)
