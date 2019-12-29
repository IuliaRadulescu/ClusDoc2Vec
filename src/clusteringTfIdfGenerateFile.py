import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import mongoConnect

mongoDb = mongoConnect.MongoDBConnector('ARXIVRobert6Clusters')

arxivRecordsCursor = mongoDb.getRecords("documents", {}, {"_id":1, "lemmatizedWords": 1, "category": 1})

docs = []
docToTf = {}
categories = {}

k = 0
for document in arxivRecordsCursor:
	docs.append(' '.join(document["lemmatizedWords"]))
	if (document["category"] in categories):
		docToTf[str(document['_id'])] = categories[document["category"]]
	else:
		categories[document["category"]] = k
		docToTf[str(document['_id'])] = k
		k = k+1

#instantiate CountVectorizer()
cv = CountVectorizer()
 
# this steps generates word counts for the words in your docs
word_count_vector = cv.fit_transform(docs)

print(np.shape(word_count_vector))

tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
 
# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(word_count_vector)

X = tf_idf_vector.todense()

np.savetxt('RobertARXIV_TFIDF_lemmatized_6_clusters.txt', X, delimiter=',', fmt='%s')

