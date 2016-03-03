# sentence_similarity

Do not add remove = ('headers', 'footers', 'quotes')
train = fetch_20newsgroups(shuffle=True, random_state=1, subset='train', remove=('headers', 'footers', 'quotes'))
Then tfidf result will be extremly bad.
