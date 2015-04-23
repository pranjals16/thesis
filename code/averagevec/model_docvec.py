#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ****** Read the two training sets and the test set
#
from word2vec import Word2Vec,Sent2Vec, LineSentence
import pandas as pd
import chardet
import math
import os,sys
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import word2vec, doc2vec
from gensim import models
import cPickle
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn import svm
from nltk import word_tokenize
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# ****** Define functions to create average word vectors
#
np.set_printoptions(threshold=np.nan)

def n_containing(word, decoded):
	return sum(1 for doc in decoded if word in doc)
	
def idf(word, decoded):
	return math.log(len(decoded)/(1+n_containing(word,decoded)))
	
	
def makeFeatureVec(words, model,num_features,counter,traintest):
	featureVec = np.zeros((num_features,),dtype="float32")
	#
	nwords = 0.
	#
	'''for word in words:
		print word.encode('utf-8')
		if word in index2word_set:
			temp = np.zeros((num_features+new_feature,),dtype="float32")
			temp[0:500]=model[word]
			temp[500:]=X[int(counter)]
			featureVec = np.add(featureVec,temp)
			nwords = nwords + 1.
	#'''
	if(traintest=='false'):
		featureVec[0:num_features]=model["SENT_"+str(int(counter))]
	else:
		featureVec[0:num_features]=model["SENT_"+str(int(counter+75000))]
	# Divide the result by the number of words to get the average
	#featureVec = np.divide(featureVec,nwords)
	return featureVec


def getAvgFeatureVecs(reviews, model, num_features,traintest):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #index2word_set = set(model.index2word)
    
    for review in reviews:
       print "Review %d of %d" % (counter, len(reviews))
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model,num_features,counter,traintest)
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
	clean_reviews = []
	for review in reviews["review"]:
		clean_reviews.append(review)
	return clean_reviews



if __name__ == '__main__':
	train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    	test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    	unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
	print "Read %d labeled train reviews, %d labeled test reviews, " "and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size )
	
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	
	logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
	logging.info("running %s" % " ".join(sys.argv))
	
	num_features = 400    # Word vector dimensionality
	min_word_count = 1   # Minimum word count
	num_workers = 8       # Number of threads to run in parallel
	context =10          # Context window size
	
	input_file = 'data/imdb.txt'
	sentences = doc2vec.LabeledLineSentence(input_file)	
	print "Training Word2Vec model..."
	model = doc2vec.Doc2Vec(sentences, workers=num_workers,size=num_features, min_count = min_word_count, window = context)
	model.init_sims(replace=True)
	model_name = "500features_10minwords_6context_docvec"
	model.save(model_name)
	model=doc2vec.Doc2Vec.load("500features_10minwords_6context_docvec")
	
	traintest='false'
	print "Creating average feature vecs for training reviews"
	trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features,traintest)
	cPickle.dump(trainDataVecs, open('save_train_docvec.p', 'wb'))
	#trainDataVecs = cPickle.load(open('save_train_docvec.p', 'rb'))
	
	traintest='true'
	print "Creating average feature vecs for testing reviews"
	testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features,traintest)
	cPickle.dump(trainDataVecs, open('save_test_docvec.p', 'wb'))
	#testDataVecs = cPickle.load(open('save_test_docvec.p', 'rb'))
	#trainDataVecs_new= SelectKBest(f_classif, k=1000).fit_transform(trainDataVecs, train["sentiment"])
	
	######################              SVM				####################
	print "Fitting a SVM classifier to labeled training data..."
	clf = svm.LinearSVC()
	clf.fit(trainDataVecs, train["sentiment"])
	print clf.score(testDataVecs,test["sentiment"])
