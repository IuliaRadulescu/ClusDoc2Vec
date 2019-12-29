import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mongoConnect
import evaluation_measures

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

for label in labels:
	evaluationDict[label] = {}

files = ['datasets/RobertARXIV_DOC2VEC_lemmatizedWords_16_dim_6_clusters_no_stop.txt', 'datasets/RobertARXIV_DOC2VEC_porterStemmedWords_16_dim_6_clusters_no_stop.txt', 'datasets/RobertARXIV_DOC2VEC_lancasterStemmedWords_16_dim_6_clusters_no_stop.txt']

# files = ['datasets/RobertARXIV_DOC2VEC_16_dim_6_clusters.txt', 'datasets/RobertARXIV_DOC2VEC_porterStemmedWords_16_dim_6_clusters.txt', 'datasets/RobertARXIV_DOC2VEC_lancasterStemmedWords_16_dim_6_clusters.txt']


for file in files:

	# read from file
	dataset = []

	with open(file) as f:
		content = f.readlines()

	content = [l.strip() for l in content]

	for l in content:
		listOfCoords = []
		aux = l.split(',')
		for dim in range(len(aux)):
			listOfCoords.append(float(aux[dim]))
		dataset.append(listOfCoords)

	#plot distance graph

	ns = 4
	nbrs = NearestNeighbors(n_neighbors=ns, metric='cosine').fit(dataset)
	# nbrs = NearestNeighbors(n_neighbors=ns).fit(dataset)
	distances, indices = nbrs.kneighbors(dataset)
	distanceDec = sorted(distances[:, ns - 1], reverse=True)

	distanceDec = np.array(distanceDec)
	maxSlopeIdx = np.argmax(distanceDec[:-1] - distanceDec[1:])

	# print("max slope " + str(distanceDec[maxSlopeIdx]))

	print("Metrici DBSCAN")

	clustering = DBSCAN(eps=0.29, metric='cosine', min_samples=320, n_jobs=8).fit(dataset)

	# clustering = DBSCAN(eps=distanceDec[maxSlopeIdx], min_samples=3, n_jobs=8).fit(dataset)

	k = 0
	for label in clustering.labels_:
		point2cluster[k] = label
		k = k + 1
		for c in evaluationDict:
			evaluationDict[c][label] = 0

	for point in point2cluster:
		evaluationDict[point2class[point]][point2cluster[point]] += 1

	print(evaluation_measures.adj_rand_index(evaluationDict))

	matriceContingenta = evaluation_measures.construct_cont_table(evaluationDict)

	print(matriceContingenta)

	np.savetxt('matriceContingenta.txt', matriceContingenta, delimiter=',', fmt='%s')

	print(metrics.homogeneity_score(labels, clustering.labels_))
	print(metrics.completeness_score(labels, clustering.labels_))
	print(metrics.v_measure_score(labels, clustering.labels_))
	print(metrics.adjusted_rand_score(labels, clustering.labels_))

	print("=======================================================================")




