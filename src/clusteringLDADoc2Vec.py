import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
from collections import defaultdict
from sklearn import metrics
import mongoConnect

mongoDb = mongoConnect.MongoDBConnector('ARXIVRobert6Clusters')

arxivRecordsCursor = mongoDb.getRecords("documents", {}, {"_id":1, "category": 1})

docs = []
categories = {}

evaluationDict = {}
point2cluster = {}
point2class = {}

k = 0
pointId = 0

for document in arxivRecordsCursor:

	if (document["category"] in categories):
		point2class[pointId] = categories[document["category"]]
	else:
		categories[document["category"]] = k
		point2class[pointId] = k
	pointId = pointId + 1

	k = k+1

labels = np.array(list(point2class.values()))

files = ['datasets/RobertARXIV_DOC2VEC_lemmatizedWords_16_dim_6_clusters_no_stop.txt', 'datasets/RobertARXIV_DOC2VEC_porterStemmedWords_16_dim_6_clusters_no_stop.txt', 'datasets/RobertARXIV_DOC2VEC_lancasterStemmedWords_16_dim_6_clusters_no_stop.txt']

for file in files:

	dataset = []

	with open(file) as f:
		content = f.readlines()

	content = [l.strip() for l in content]

	for l in content:
		listOfCoords = []
		aux = l.split(',')
		for dim in range(len(aux)):
			listOfCoords.append(float(aux[dim]))
		# normalize values
		normalizedListOfCoords = [ (x - min(listOfCoords))/(max(listOfCoords) - min(listOfCoords)) for x in listOfCoords]
		dataset.append(normalizedListOfCoords)

	X = np.array(dataset)

	# Create and fit the LDA model
	lda = LDA(n_components=6, n_jobs=-1)
	lda.fit(X)# Print the topics found by the LDA model
	# print("Topics found via LDA:")
	# print_topics(lda, cv, 10)

	documentTopicDistr = lda.transform(X)
	documentTopicDistr = np.array(documentTopicDistr)

	lda_labels = np.argmax(documentTopicDistr, axis=1)

	print("Metrici LDA " + file)

	print(metrics.homogeneity_score(labels, lda_labels))
	print(metrics.completeness_score(labels, lda_labels))
	print(metrics.v_measure_score(labels, lda_labels))
	print(metrics.adjusted_rand_score(labels, lda_labels))
	print(metrics.adjusted_mutual_info_score(labels, lda_labels))

	print("=============================================")

