#!/usr/bin/env python


# ****** Read the two training sets and the test set
#
import pandas as pd
import os
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
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    index2word_set = set(model.index2word)
    #
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
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
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       #
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)
       #
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



if __name__ == '__main__':

	train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
	test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
	unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
	print "Read %d labeled train reviews, %d labeled test reviews, " "and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size )
	#sentences=cPickle.load(open('sentences.p', 'rb'))
	#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
	# Set values for various parameters
	num_features = 300    # Word vector dimensionality
	min_word_count = 20   # Minimum word count
	num_workers = 16       # Number of threads to run in parallel
	context = 10          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words
	# Initialize and train the model (this will take some time)
	print "Training Word2Vec model..."
	'''
	model = word2vec.Word2Vec(sentences, workers=num_workers,size=num_features, min_count = min_word_count,window = context, sample = downsampling, seed=1)
	model.init_sims(replace=True)
	model_name = "300features_40minwords_10context"
	model.save(model_name)
	'''
	model = word2vec.Word2Vec.load("300features_40minwords_10context")
	
	print "Creating average feature vecs for training reviews"
	#trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features )
	#cPickle.dump(trainDataVecs, open('save_train.p', 'wb'))
	trainDataVecs = cPickle.load(open('save_train.p', 'rb'))

	print "Creating average feature vecs for test reviews"
	#testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features )
	#cPickle.dump(testDataVecs, open('save_test.p', 'wb'))
	testDataVecs = cPickle.load(open('save_test.p', 'rb'))
	
	y_train = train["sentiment"]  
	y_test=test["sentiment"]
	'''
	print("Cleaning and parsing movie reviews...\n") 
	traindata = []
	for i in range( 0, len(train["review"])):
		traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], False)))
	testdata = []
	for i in range(0,len(test["review"])):
		testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], False)))
	
	print ('vectorizing... ',) 
	tfv = TfidfVectorizer(min_df=2,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=2,smooth_idf=1,sublinear_tf=1,stop_words='english')
	X_all = traindata + testdata
	lentrain = len(traindata)

	print ("fitting pipeline... ",)
	tfv.fit(X_all)
	X_all = tfv.transform(X_all)
	#cPickle.dump(X_all, open('X_all.p', 'wb'))
	'''
	lentrain=25000
	X_all=cPickle.load(open('X_all.p', 'rb'))
	X = X_all[:lentrain]
	X_test = X_all[lentrain:]
	newTrain = []
	for d,t in zip(trainDataVecs, X):
		temp=np.zeros((num_features+t.shape[1]),dtype="float32")
		temp[0:num_features]=d
		temp[num_features:]=t.toarray()
		newTrain.append(temp)
	newTest = []
	for d,t in zip(testDataVecs, X_test):
		temp=np.zeros((num_features)+t.shape[1],dtype="float32")
		temp[0:num_features]=d
		temp[num_features:]=t
		newTest.append(temp)
		
	#newTrain = np.hstack((trainDataVecs, X.toarray()))
	#newTest = np.hstack((testDataVecs, X_test.toarray()))
	#kbest= SelectKBest(f_classif, k=20000).fit(X, y_train)
	#newTrain=kbest.transform(X)
	#newTest=kbest.transform(X_test)
	model = LogisticRegression(penalty='l2', dual=True, tol=0.0001,C=14, fit_intercept=True, intercept_scaling=1,class_weight=None, random_state=None)
	print "Fitting a LogisticRegression classifier to labeled training data..."
	model.fit(newTrain,y_train)
	print model.score(newTest,y_test)
	#------------------------------------------------------------------------------------------
	######################              SVM				####################
	print "Fitting a SVM classifier to labeled training data..."
	clf = svm.LinearSVC()
	clf.fit(newTrain, y_train)
	print clf.score(newTest,y_test)
	#------------------------------------------------------------------------------------------
