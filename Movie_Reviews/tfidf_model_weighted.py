#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ****** Read the two training sets and the test set
#
import pandas as pd
import chardet
import math
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import word2vec
from gensim import models
from sklearn.linear_model import Ridge
import cPickle
from sklearn import svm,grid_search
from nltk import word_tokenize
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.lda import LDA

# ****** Define functions to create average word vectors
#
np.set_printoptions(threshold=np.nan)

def n_containing(word, decoded):
	return sum(1 for doc in decoded if word in doc)
	
def idf(word, decoded):
	return math.log(len(decoded)/(1+n_containing(word,decoded)))
	
def tokenize(text):
	tokens = word_tokenize(text)
	stems = []
	for item in tokens:
		stems.append(item)
	return stems
	
def makeFeatureVec(words, model,num_features,index2word_set,X,counter,feature_names,idf_scores):
	new_feature=X.shape[1]
	featureVec = np.zeros((num_features+new_feature,),dtype="float32")
	#
	nwords = 0.
	#
	#stop_words=set(line.strip() for line in open('stopwords_hi.txt'))
	
	for word in words:
		if word in index2word_set:
			temp = np.zeros((num_features+new_feature,),dtype="float32")
			#if(word in idf_scores[int(counter)]):
			if (word in feature_names and feature_names.index(word)<len(idf_scores)):
				temp[0:num_features]=np.multiply(model[word],idf_scores[feature_names.index(word)])
			#	temp[0:num_features]=np.multiply(model[word],idf_scores[int(counter)][word])
			else:
				temp[0:num_features]=model[word]
			temp[num_features:]=X[int(counter)]
			#print X[counter][word]
			featureVec = np.add(featureVec,temp)
			nwords = nwords + 1.
	#
	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)
	#print len(index2word_set2)
	return featureVec


def getAvgFeatureVecs(reviews, model, num_features,X,feature_names,idf_scores):
    counter = 0.
    new_feature=X.shape[1]
    reviewFeatureVecs = np.zeros((len(reviews),num_features+new_feature),dtype="float32")
    index2word_set = set(model.index2word)
    
    for review in reviews:
       #print "Review %d of %d" % (counter, len(reviews))
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model,num_features,index2word_set,X,counter,feature_names,idf_scores)
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
	clean_reviews = []
	for review in reviews["review"]:
		raw=review.decode('utf-8')
		tokens = word_tokenize(raw)
		clean_reviews.append(tokens)
	return clean_reviews



if __name__ == '__main__':
	train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'new_movie_reviews.tsv'), header=0, delimiter="\t", quoting=3 )
	print "Read %d labeled train reviews\n" % (train["review"].size)

	num_features = 600    # Word vector dimensionality
	min_word_count = 1   # Minimum word count
	num_workers = 8       # Number of threads to run in parallel
	context =6          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words
	sentences = word2vec.Text8Corpus('new_movie_reviews2.txt')
	print "Training Word2Vec model..."
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
	'''
	model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count,	window = context, sample = downsampling, seed=1)
	model.init_sims(replace=True)

	model_name = "300features_40minwords_10context"
	model.save(model_name)
	'''
	model = word2vec.Word2Vec.load("300features_40minwords_10context")
	'''
	f=open('new_movie_reviews2.txt')
	corpus=[]
	for line in f:
		line= line.strip()
		corpus.append(line)
	decoded = [x.decode(chardet.detect(x)['encoding']) for x in (corpus)]
	
	count_vectorizer = CountVectorizer(max_features=25000,tokenizer=tokenize,strip_accents='unicode')
	count_vectorizer.fit_transform(decoded)
	freq_term_matrix = count_vectorizer.transform(decoded)
	tfidf = TfidfTransformer(norm="l2")
	tfidf.fit(freq_term_matrix)
	temp= count_vectorizer.get_feature_names()
	feature_names=[]
	for word in temp:
		feature_names.append(word.encode('utf-8'))
	
	vectorizer = TfidfVectorizer(tokenizer=tokenize,use_idf=True,min_df=0.001,strip_accents='unicode')
	X=vectorizer.fit_transform(decoded).toarray()

	temp= vectorizer.get_feature_names()
	feature_names=[]
	for word in temp:
		feature_names.append(word)
	#idf_scores=[]
	#for doc in decoded:
	#	idf_scores.append({word: idf(word, decoded) for word in doc.replace(',',' ,').split()})
	idf_scores=tfidf.idf_
	'''
	print "Creating average feature vecs for training reviews"
	#trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features,X,feature_names,idf_scores)
	#cPickle.dump(trainDataVecs, open('save_train.p', 'wb'))
	trainDataVecs = cPickle.load(open('save_train.p', 'rb'))
	print trainDataVecs.shape
	#trainDataVecs_new= SelectKBest(f_classif, k=4000).fit_transform(trainDataVecs, train["sentiment"])
	trainDataVecs_new= LDA().fit_transform(trainDataVecs, train["sentiment"])
	print trainDataVecs_new.shape
	'''
	NUM_TOPICS = 3
	X=trainDataVecs_new
	y=train["sentiment"]
	kmeans = KMeans(NUM_TOPICS).fit(X)
	colors = ["b", "g", "r", "m", "c"]
	for i in range(X.shape[0]):
		plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=20)    
	plt.show()
	'''
	'''
	######################              SVM				####################
	print "Fitting a LogisticRegression to labeled training data..."
	clf = LogisticRegression()
	clf.fit(trainDataVecs, train["sentiment"])
	scores= cross_validation.cross_val_score(clf, trainDataVecs,train["sentiment"], cv=20)
	print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	'''
	######################              SVM				####################
	print "Fitting a Linear SVM classifier to labeled training data..."
	clf = svm.LinearSVC()
	clf.fit(trainDataVecs_new, train["sentiment"])
	scores= cross_validation.cross_val_score(clf, trainDataVecs_new,train["sentiment"], cv=20)
	print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	#cscore=clf.decision_function(trainDataVecs_new)
	#predictions=clf.predict(trainDataVecs_new)
	#print cscore
	#print predictions
	print "Fitting a Linear SVM classifier to labeled training data..."
	clf = svm.LinearSVC(C=0.9)
	clf.fit(trainDataVecs, train["sentiment"])
	scores= cross_validation.cross_val_score(clf, trainDataVecs,train["sentiment"], cv=20)
	print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
