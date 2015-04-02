#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ****** Read the two training sets and the test set
#
import pandas as pd
import chardet
import math
import os,sys
import logging
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import word2vec
from gensim import models
from sklearn.ensemble import RandomForestClassifier
import cPickle
from sklearn import svm
from KaggleWord2VecUtility import KaggleWord2VecUtility
from nltk import word_tokenize
from sklearn import cross_validation
from sklearn import naive_bayes
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

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
	
def makeFeatureVec(words, model,num_features,index2word_set,X,counter,feature_names,idf_scores_pos,idf_scores_neg):
	new_feature=X.shape[1]
	featureVec = np.zeros((num_features+new_feature,),dtype="float32")
	#
	nwords = 0.
	size=350
	#
	#stop_words=set(line.strip() for line in open('stopwords_hi.txt'))
	
	for word in words:
		if word in index2word_set:
			temp = np.zeros((num_features+new_feature,),dtype="float32")
			if(int(counter)<size and word in idf_scores_pos[int(counter)]):
				temp[0:num_features]=np.multiply(model[word],idf_scores_pos[int(counter)][word])
				
			elif(int(counter)>=size and word in idf_scores_neg[size-int(counter)]):
				temp[0:num_features]=np.multiply(model[word],idf_scores_neg[size-int(counter)][word])
				
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


def getAvgFeatureVecs(reviews, model, num_features,X,feature_names,idf_scores_pos,idf_scores_neg):
    counter = 0.
    new_feature=X.shape[1]
    reviewFeatureVecs = np.zeros((len(reviews),num_features+new_feature),dtype="float32")
    index2word_set = set(model.index2word)
    
    for review in reviews:
       #print "Review %d of %d" % (counter, len(reviews))
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model,num_features,index2word_set,X,counter,feature_names,idf_scores_pos,idf_scores_neg)
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
	train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'iiit.tsv'), header=0, delimiter="\t", quoting=3 )
	print "Read %d labeled train reviews\n" % (train["review"].size)
	logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
	logging.info("running %s" % " ".join(sys.argv))
	
	num_features =500    # Word vector dimensionality
	min_word_count = 1   # Minimum word count
	num_workers = 8       # Number of threads to run in parallel
	context =5          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words
	sentences = word2vec.Text8Corpus('data/hindi_review_product.txt')
	#sentences2 = word2vec.Text8Corpus('../../hindi')
	print "Training Word2Vec model..."
	model = word2vec.Word2Vec(sentences, workers=num_workers,size=num_features, min_count = min_word_count,window = context, seed=1)
	#model.train(sentences2)
	model.init_sims(replace=True)

	model_name = "300features_40minwords_10context"
	model.save(model_name)

	f=open('data/hindi_review_product.txt')
	corpus=[]
	for line in f:
		line= line.strip()
		corpus.append(line)
	decoded = [x.decode(chardet.detect(x)['encoding']) for x in (corpus)]

	vectorizer = TfidfVectorizer(tokenizer=tokenize,use_idf=True,max_df=0.3,min_df=0.0001,strip_accents='unicode')
	X=vectorizer.fit_transform(decoded).toarray()

	temp= vectorizer.get_feature_names()
	feature_names=[]
	for word in temp:
		feature_names.append(word)
	
	############################ idf for pos+neg  #########################################################
	f=open('data/ProductReviews (350+350)/Positive.txt')
	#f=open('data/HindiPolarityLabeledCorpora/Positive.txt')
	corpus=[]
	for line in f:
		line= line.strip()
		corpus.append(line)
	decoded = [x.decode(chardet.detect(x)['encoding']) for x in (corpus)]
	
	idf_scores_pos=[]
	for doc in decoded:
		idf_scores_pos.append({word: idf(word, decoded) for word in doc.replace(',',' ,').split()})
		
	f=open('data/ProductReviews (350+350)/Negative.txt')
	#f=open('data/HindiPolarityLabeledCorpora/Negative.txt')
	corpus=[]
	decoded=[]
	for line in f:
		line= line.strip()
		corpus.append(line)
	decoded = [x.decode(chardet.detect(x)['encoding']) for x in (corpus)]
	
	idf_scores_neg=[]
	for doc in decoded:
		idf_scores_neg.append({word: idf(word, decoded) for word in doc.replace(',',' ,').split()})
	
	#############################################################################################################
	print "Creating average feature vecs for training reviews"
	trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features,X,feature_names,idf_scores_pos,idf_scores_neg)

	######################              SVM				####################
	print "Fitting a SVM classifier to labeled training data..."
	clf = svm.LinearSVC()
	clf.fit(trainDataVecs, train["sentiment"])
	scores= cross_validation.cross_val_score(clf, trainDataVecs,train["sentiment"], cv=20)
	print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
