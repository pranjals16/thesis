#!/usr/bin/env python
#
import pandas as pd
import os
import chardet
import math
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
import cPickle
from sklearn import svm
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn import naive_bayes
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

#
def n_containing(word, decoded):
	return sum(1 for doc in decoded if word in doc)
	
def idf(word, decoded,count):
	print count
	return math.log(len(decoded)/(1+n_containing(word,decoded)))
	
def makeFeatureVec(words, model, num_features,counter):
	featureVec = np.zeros((num_features,),dtype="float32")
	#
	nwords = 0.
	#
	index2word_set = set(model.index2word)
	#
	for word in words:
		if word in index2word_set:
			temp = np.zeros((num_features),dtype="float32")
			temp[0:num_features]=model[word]
			nwords = nwords + 1.
			featureVec = np.add(featureVec,temp)
	#
	featureVec = np.divide(featureVec,nwords)
	return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    #
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    for review in reviews:
       #
       print "Review %d of %d" % (counter, len(reviews))
       #
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features,counter)
       #
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



if __name__ == '__main__':

	total = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'trainData_mp3.tsv'), header=0, delimiter="\t", quoting=3 )
	train, test = train_test_split(total, train_size=0.80, random_state=1)
	Train = pd.DataFrame(train, columns=total.columns)
	Test = pd.DataFrame(test, columns=total.columns)
	print "Read %d labeled train reviews, %d labeled test reviews \n" % (Train["review"].size,Test["review"].size)

	# Load the punkt tokenizer
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	# ****** Split the labeled and unlabeled training sets into clean sentences
	#
	sentences = []  # Initialize an empty list of sentences

	print "Parsing sentences from training set"
	for review in Train["review"]:
	   sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

	print "Parsing sentences from unlabeled set"
	for review in Test["review"]:
	   sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
	# ****** Set parameters and train the word2vec model
	#
	# Import the built-in logging module and configure it so that Word2Vec
	# creates nice output messages
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
	# Set values for various parameters
	num_features = 500    # Word vector dimensionality
	min_word_count = 10   # Minimum word count
	num_workers = 8       # Number of threads to run in parallel
	context = 10          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words
	# Initialize and train the model (this will take some time)
	print "Training Word2Vec model..."
	model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling, seed=1,negative=10)
	model.init_sims(replace=True)
	model_name = "500features_10minwords_5context_mp3"
	model.save(model_name)
	#model = word2vec.Word2Vec.load("500features_10minwords_5context_mp3")
	# ****** Create average vectors for the training and test sets
	#
	print "Creating average feature vecs for training reviews"
	trainDataVecs = getAvgFeatureVecs( getCleanReviews(Train), model, num_features )
	cPickle.dump(trainDataVecs, open('save_train_mp3.p', 'wb'))
	#trainDataVecs = cPickle.load(open('save_train.p', 'rb'))

	print "Creating average feature vecs for test reviews"
	testDataVecs = getAvgFeatureVecs( getCleanReviews(Test), model, num_features)
	cPickle.dump(testDataVecs, open('save_test_mp3.p', 'wb'))
	#testDataVecs = cPickle.load(open('save_test.p', 'rb'))
	#------------------------------------------------------------------------------------------
	######################              SVM				####################
	trainDataVecs = Imputer(strategy='mean').fit_transform(trainDataVecs)
	testDataVecs = Imputer(strategy='mean').fit_transform(testDataVecs)
	Y_train=Train["sentiment"].astype(int)
	Y_test=Test["sentiment"].astype(int)
	print "Fitting a SVM classifier to labeled training data..."
	clf = svm.LinearSVC()
	clf.fit(trainDataVecs, Y_train)
	print clf.score(testDataVecs,Y_test)
	#------------------------------------------------------------------------------------------
