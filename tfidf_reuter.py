from nltk.corpus import reuters 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from time import time


n_topics = 60
categories = ["acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade"]
docs = reuters.fileids(categories);
## Only keep documents that have one category
docs = list(filter(lambda doc: len(reuters.categories(doc)) == 1, docs))

train = list(filter(lambda doc: doc.startswith("train"), docs));
test = list(filter(lambda doc: doc.startswith("test"), docs));

train_docs = list(reuters.raw(doc) for doc in train)
test_docs = list(reuters.raw(doc) for doc in test)

train_target = list(reuters.categories(doc) for doc in train)
test_target = list(reuters.categories(doc) for doc in test)

## Using tfidf
vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 5, stop_words = 'english')
vectors = vectorizer.fit_transform(train_docs)
vectors_test = vectorizer.transform(test_docs)
print vectors.shape

t0 = time()
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(vectors, train_target)
pred = neigh.predict(vectors_test)
# # print float(list(test.target[:100]==r).count(True))/len(r)
print "tf-idf result: "
print metrics.accuracy_score(test_target,pred)
print("done in %0.3fs" % (time() - t0))


