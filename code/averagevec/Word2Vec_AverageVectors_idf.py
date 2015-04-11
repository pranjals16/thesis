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

#
def n_containing(word, decoded):
	return sum(1 for doc in decoded if word in doc)
	
def idf(word, decoded,count):
	print count
	return math.log(len(decoded)/(1+n_containing(word,decoded)))
	
def makeFeatureVec(words, model, num_features,counter,idf_score,feature_names):
	featureVec = np.zeros((num_features,),dtype="float32")
	#
	nwords = 0.
	#
	index2word_set = set(model.index2word)
	#
	for word in words:
		if word in index2word_set:
			temp = np.zeros((num_features),dtype="float32")
			#if word in feature_names:
			if(word in idf_score[int(counter)]):
				temp[0:num_features]=np.multiply(model[word],idf_score[int(counter)][word])
			else:
				temp[0:num_features]=model[word]
			nwords = nwords + 1.
			featureVec = np.add(featureVec,temp)
	#
	featureVec = np.divide(featureVec,nwords)
	return featureVec


def getAvgFeatureVecs(reviews, model, num_features,idf_score,feature_names):
    counter = 0.
    #
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    for review in reviews:
       #
       print "Review %d of %d" % (counter, len(reviews))
       #
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features,counter,idf_score,feature_names)
       #
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



if __name__ == '__main__':

	# Read data from files
	train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
	test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
	unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )

	# Verify the number of reviews that were read (100,000 in total)
	print "Read %d labeled train reviews, %d labeled test reviews, " \
	"and %d unlabeled reviews\n" % (train["review"].size,
	test["review"].size, unlabeled_train["review"].size )

	# Load the punkt tokenizer
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	# ****** Split the labeled and unlabeled training sets into clean sentences
	#
	sentences = []  # Initialize an empty list of sentences

	print "Parsing sentences from training set"
	for review in train["review"]:
	   sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

	print "Parsing sentences from unlabeled set"
	for review in unlabeled_train["review"]:
	   sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
	# ****** Set parameters and train the word2vec model
	#
	# Import the built-in logging module and configure it so that Word2Vec
	# creates nice output messages
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
	   level=logging.INFO)
	# Set values for various parameters
	num_features = 500    # Word vector dimensionality
	min_word_count = 20   # Minimum word count
	num_workers = 8       # Number of threads to run in parallel
	context = 10          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words
	# Initialize and train the model (this will take some time)
	print "Training Word2Vec model..."
	model = word2vec.Word2Vec(sentences, workers=num_workers, \
		      size=num_features, min_count = min_word_count, \
		      window = context, sample = downsampling, seed=1)

	# If you don't plan to train the model any further, calling
	# init_sims will make the model much more memory-efficient.
	model.init_sims(replace=True)

	# It can be helpful to create a meaningful model name and
	# save the model for later use. You can load it later using Word2Vec.load()
	model_name = "300features_40minwords_10context"
	model.save(model_name)
	#model = word2vec.Word2Vec.load("300features_40minwords_10context")
	# ****** Create average vectors for the training and test sets
	#

	f=[]
	for review in train["review"]:
		f.append(review)
	corpus=[]
	for line in f:
		line= line.strip()
		corpus.append(line)
	decoded = [x for x in (corpus)]
	print "Decoded Done!!"
	count_vectorizer = CountVectorizer(max_features=20000,stop_words="english")
	count_vectorizer.fit_transform(decoded)
	freq_term_matrix = count_vectorizer.transform(decoded)
	tfidf = TfidfTransformer(norm="l2")
	tfidf.fit(freq_term_matrix)
	temp= count_vectorizer.get_feature_names()
	feature_names=[]
	for word in temp:
		feature_names.append(word)
	#idf_score=tfidf.idf_
	'''
	idf_scores=[]
	count=0
	for doc in decoded:
		count=count+1
		idf_scores.append({word: idf(word, decoded,count) for word in doc.replace(',',' ,').split()})
	cPickle.dump(idf_scores, open('idf_score.p', 'wb'))
	'''
	idf_score = cPickle.load(open('idf_score.p', 'rb'))
	print "Creating average feature vecs for training reviews"
	trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features,idf_score,feature_names )
	#cPickle.dump(trainDataVecs, open('save_train.p', 'wb'))
	#trainDataVecs = cPickle.load(open('save_train.p', 'rb'))

	print "Creating average feature vecs for test reviews"
	testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features,idf_score,feature_names )
	#cPickle.dump(testDataVecs, open('save_test.p', 'wb'))
	#testDataVecs = cPickle.load(open('save_test.p', 'rb'))
	'''
	######################              Naive Bayes				####################
	print "Fitting a Naive Bayes classifier to labeled training data..."
	clf = naive_bayes.GaussianNB()
	clf.fit(trainDataVecs, train["sentiment"])
	print clf.score(testDataVecs,test["sentiment"])
	#------------------------------------------------------------------------------------------
	'''
	#------------------------------------------------------------------------------------------
	######################              SVM				####################
	print "Fitting a SVM classifier to labeled training data..."
	clf = svm.LinearSVC()
	clf.fit(trainDataVecs, train["sentiment"])
	print clf.score(testDataVecs,test["sentiment"])
	#------------------------------------------------------------------------------------------
