import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import defaultdict
import mongoConnect

def each_document_as_topics(model, count_vectorizer, docsAsListOfWords, n_top_words):
	words = count_vectorizer.get_feature_names()
	wordsByTopics = {}
	for topic_idx, topic in enumerate(model.components_):
		wordsByTopics[topic_idx] = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

	reducedDimsDocs = defaultdict(list)

	for doc_idx, doc in enumerate(docsAsListOfWords):
		for topic in wordsByTopics:
			for word in wordsByTopics[topic]:
				reducedDimsDocs[doc_idx].append(word)

	return list(reducedDimsDocs.values())

mongoDb = mongoConnect.MongoDBConnector('ARXIVRobert6Clusters')

arxivRecordsCursor = mongoDb.getRecords("documents", {}, {"_id":1, "lemmatizedWords": 1, "category": 1})

docs = []
docToTf = {}
categories = {}

docsAsListOfWords = []

for document in arxivRecordsCursor:
	docs.append(' '.join(document["lemmatizedWords"]))
	docsAsListOfWords.append(document["lemmatizedWords"])

#instantiate CountVectorizer()
cv = CountVectorizer(stop_words='english')
 
# this steps generates word counts for the words in your docs
word_count_vector = cv.fit_transform(docs)

# Create and fit the LDA model
lda = LDA(n_components=6, n_jobs=-1)
lda.fit(word_count_vector)


print("docsAsListOfWords " + str(np.shape(docsAsListOfWords)))

docs = each_document_as_topics(lda, cv, docsAsListOfWords, 20)

taggedDocuments = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
model = Doc2Vec(vector_size=4, window=3, min_count=0, workers=4, epochs=200)

model.build_vocab(taggedDocuments)

model.train(taggedDocuments, total_examples=len(taggedDocuments), epochs=200)

X = []

for doc in docs:
	X.append(model.infer_vector(doc))

X = np.array(X)

np.savetxt('RobertARXIV_DOC2VEC_4_dim_6_clusters_LDA.txt', X, delimiter=',', fmt='%s')