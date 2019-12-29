import nltk
import os
import time
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet

import mongoConnect

mongoDb = mongoConnect.MongoDBConnector('ARXIVRobert6Clusters')

porter = PorterStemmer()
lancaster=LancasterStemmer()
lemmatizer = WordNetLemmatizer() 

arxivRecordsCursor = mongoDb.getRecords("documents", {}, {"_id":1, "summary": 1})

for document in arxivRecordsCursor:

  words = []
  porterStemmedWords = []
  lancasterStemmedWords = []
  lemmatizedWords = []

  summary = document["summary"]
  tokenizer = RegexpTokenizer(r'\w+')
  words = tokenizer.tokenize(summary)

  for word in words:
    porterStemmedWords.append(porter.stem(word))
    lancasterStemmedWords.append(lancaster.stem(word))
    lemmatizedWords.append(lemmatizer.lemmatize(word))

  updateFields = {
    "words": words,
    "porterStemmedWords": porterStemmedWords,
    "lancasterStemmedWords": lancasterStemmedWords,
    "lemmatizedWords": lemmatizedWords
  }

  mongoDb.update("documents", {"_id": document["_id"]}, {"$set": updateFields}, False, True)
