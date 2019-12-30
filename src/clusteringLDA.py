import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from collections import defaultdict
from sklearn import metrics
import seaborn as sns
import mongoConnect

def print_topics(model, count_vectorizer, n_top_words):
	words = count_vectorizer.get_feature_names()
	for topic_idx, topic in enumerate(model.components_):
		print("\nTopic #%d:" % topic_idx)
		print(" ".join([words[i]
						for i in topic.argsort()[:-n_top_words - 1:-1]]))

def each_document_as_topics(model, count_vectorizer, docsAsListOfWords, n_top_words):
	words = count_vectorizer.get_feature_names()
	wordsByTopics = {}
	for topic_idx, topic in enumerate(model.components_):
		wordsByTopics[topic_idx] = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

	reducedDimsDocs = defaultdict(list)

	for doc_idx, doc in enumerate(docsAsListOfWords):
		for topic in wordsByTopics:
			for word in wordsByTopics[topic]:
				if word in doc:
					reducedDimsDocs[doc_idx].append(word)

	values = list(reducedDimsDocs.values())


def plot_10_most_common_words(count_data, count_vectorizer):

	sns.set_style('whitegrid')

	words = count_vectorizer.get_feature_names()
	total_counts = np.zeros(len(words))
	for t in count_data:
		total_counts+=t.toarray()[0]
	
	count_dict = (zip(words, total_counts))
	count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
	words = [w[0] for w in count_dict]
	counts = [w[1] for w in count_dict]
	x_pos = np.arange(len(words)) 
	
	plt.figure(2, figsize=(15, 15/1.6180))
	plt.subplot(title='ARXIV dataset: 10 most common words')
	sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
	sns.barplot(x_pos, counts, palette='husl')
	plt.xticks(x_pos, words, rotation=90) 
	plt.xlabel('words')
	plt.ylabel('counts')
	plt.savefig('top10MostCommonWords.png')

mongoDb = mongoConnect.MongoDBConnector('ARXIVRobert6Clusters')

stemmers = ["porterStemmedWords", "lancasterStemmedWords", "lemmatizedWords"]

for stemmer in stemmers:

	arxivRecordsCursor = mongoDb.getRecords("documents", {}, {"_id":1, stemmer: 1, "category": 1})

	docs = []
	docToLabel = {}
	categories = {}

	docsAsListOfWords = []

	k = 0
	for document in arxivRecordsCursor:
		docs.append(' '.join(document[stemmer]))
		docsAsListOfWords.append(document[stemmer])
		if (document["category"] in categories):
			docToLabel[str(document['_id'])] = categories[document["category"]]
		else:
			categories[document["category"]] = k
			docToLabel[str(document['_id'])] = k
			k = k+1

	labels = np.array(list(docToLabel.values()))

	#instantiate CountVectorizer()
	cv = CountVectorizer(stop_words='english')
	 
	# this steps generates word counts for the words in your docs
	word_count_vector = cv.fit_transform(docs)

	# Create and fit the LDA model
	lda = LDA(n_components=6, n_jobs=-1)
	lda.fit(word_count_vector)# Print the topics found by the LDA model
	# print("Topics found via LDA:")
	# print_topics(lda, cv, 10)

	documentTopicDistr = lda.transform(word_count_vector)
	documentTopicDistr = np.array(documentTopicDistr)

	lda_labels = np.argmax(documentTopicDistr, axis=1)

	print("Metrici LDA " + stemmer)

	print(metrics.homogeneity_score(labels, lda_labels))
	print(metrics.completeness_score(labels, lda_labels))
	print(metrics.v_measure_score(labels, lda_labels))
	print(metrics.adjusted_rand_score(labels, lda_labels))
	print(metrics.adjusted_mutual_info_score(labels, lda_labels))

	print("=============================================")

