import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import mongoConnect

mongoDb = mongoConnect.MongoDBConnector('ARXIVRobert6Clusters')

arxivRecordsCursor = mongoDb.getRecords("documents", {}, {"_id":1, "lemmatizedWords": 1, "category": 1})

docs = []

for document in arxivRecordsCursor:
    docs.append(' '.join(document["lemmatizedWords"]))

docs = docs[0:10]

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

lda = LatentDirichletAllocation(n_components=16, random_state=0)
lda.fit(X) 
t = lda.transform(X)

np.savetxt('RobertARXIV_TFIDF_lemmatized_LDA_6_clusters.txt', t, delimiter=',', fmt='%s')