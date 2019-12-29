import numpy as np
import copy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sphereclusterGit.spherecluster import spherical_kmeans
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import check_array
from sklearn import metrics
import mongoConnect

mongoDb = mongoConnect.MongoDBConnector('ARXIVRobert6Clusters')

arxivRecordsCursor = mongoDb.getRecords("documents", {}, {"_id":1, "category": 1})

docs = []
docLabel = {}
categories = {}

k = 0
for document in arxivRecordsCursor:

    if (document["category"] in categories):
        docLabel[str(document['_id'])] = categories[document["category"]]
    else:
        categories[document["category"]] = k
        docLabel[str(document['_id'])] = k
    k = k+1

labels = np.array(list(docLabel.values()))

# read from file

outputFile = open("RobertKMeansSphericalKmeansDoc2Vec.txt", "a")

files = files = ['datasets/RobertARXIV_DOC2VEC_16_dim_6_clusters.txt', 'datasets/RobertARXIV_DOC2VEC_porterStemmedWords_16_dim_6_clusters.txt', 'datasets/RobertARXIV_DOC2VEC_lancasterStemmedWords_16_dim_6_clusters.txt']

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
        dataset.append(listOfCoords)

    X = copy.deepcopy(np.array(dataset))

    print(X)

    print(np.shape(X))
    
    skm = spherical_kmeans.SphericalKMeans(n_clusters=6, init='k-means++', n_init=20)
    skm.fit(X)
    skm.labels_ = skm.labels_.tolist()

    print("Metrici spherical k-means pentru " + str(file))

    print(metrics.homogeneity_score(labels, skm.labels_))
    print(metrics.completeness_score(labels, skm.labels_))
    print(metrics.v_measure_score(labels, skm.labels_))
    print(metrics.adjusted_rand_score(labels, skm.labels_))
    print(metrics.adjusted_mutual_info_score(labels, skm.labels_))

    outputFile.write("Metrici spherical k-means pentru " + str(file) + '\n')

    outputFile.write(str(metrics.homogeneity_score(labels, skm.labels_)) + '\n')
    outputFile.write(str(metrics.completeness_score(labels, skm.labels_)) + '\n')
    outputFile.write(str(metrics.v_measure_score(labels, skm.labels_)) + '\n')
    outputFile.write(str(metrics.adjusted_rand_score(labels, skm.labels_)) + '\n')
    outputFile.write(str(metrics.adjusted_mutual_info_score(labels, skm.labels_)) + '\n')

    X = copy.deepcopy(np.array(dataset))

    print(np.shape(X))

    km = KMeans(n_clusters=6, init='k-means++', n_init=20)
    km.fit(X)
    km.labels_ = km.labels_.tolist()

    print("Metrici k-means pentru " + str(file))

    print(metrics.homogeneity_score(labels, km.labels_))
    print(metrics.completeness_score(labels, km.labels_))
    print(metrics.v_measure_score(labels, km.labels_))
    print(metrics.adjusted_rand_score(labels, km.labels_))
    print(metrics.adjusted_mutual_info_score(labels, km.labels_))

    outputFile.write("Metrici k-means pentru " + str(file) + '\n')

    outputFile.write(str(metrics.homogeneity_score(labels, km.labels_)) + '\n')
    outputFile.write(str(metrics.completeness_score(labels, km.labels_)) + '\n')
    outputFile.write(str(metrics.v_measure_score(labels, km.labels_)) + '\n')
    outputFile.write(str(metrics.adjusted_rand_score(labels, km.labels_)) + '\n')
    outputFile.write(str(metrics.adjusted_mutual_info_score(labels, km.labels_)) + '\n')

    print("===================================================================")

outputFile.close()




