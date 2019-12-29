import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
import mongoConnect
from nltk.tokenize import RegexpTokenizer

mongoDb = mongoConnect.MongoDBConnector('ARXIVRobert6Clusters')

stemmers = ["porterStemmedWords", "lancasterStemmedWords", "lemmatizedWords"]

stop_words = set(stopwords.words('english'))

for stemmer in stemmers:

	arxivRecordsCursor = mongoDb.getRecords("documents", {}, {"_id":1, stemmer: 1, "category": 1})

	docs = []

	for document in arxivRecordsCursor:
		documentWithoutStop = [w for w in document[stemmer] if not w in stop_words]
		docs.append(documentWithoutStop)

	taggedDocuments = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
	model = Doc2Vec(vector_size=16, window=3, min_count=2, workers=4, epochs=200)

	model.build_vocab(taggedDocuments)

	model.train(taggedDocuments, total_examples=len(taggedDocuments), epochs=200)

	X = []

	for doc in docs:
		X.append(model.infer_vector(doc))

	X = np.array(X)

	np.savetxt('RobertARXIV_DOC2VEC_' + stemmer + '_16_dim_6_clusters_no_stop.txt', X, delimiter=',', fmt='%s')