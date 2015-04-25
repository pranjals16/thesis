#!/usr/bin/env python
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
import cPickle
from sklearn import svm,grid_search
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn import naive_bayes
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step
		
def makeFeatureVec(words, model, num_features,index2word_set,feature_names,idf_score):
	featureVec = np.zeros((num_features,),dtype="float32")
	nwords = 0.
	
	#
	for word in words:
		if word in index2word_set:
			temp = np.zeros((num_features),dtype="float32")
			if word in feature_names.keys():
				temp[0:num_features]=np.multiply(model[word],idf_score[feature_names[word]])
			else:
				temp[0:num_features]=model[word]
			nwords = nwords + 1.
			featureVec = np.add(featureVec,model[word])
	#
	featureVec = np.divide(featureVec,nwords)
	return featureVec


def getAvgFeatureVecs(reviews, model, num_features,feature_names,idf_score):
    counter = 0.
    #
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    index2word_set = set(model.index2word)
    #
    for review in reviews:
       #
       if counter%100. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features,index2word_set,feature_names,idf_score)
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



if __name__ == '__main__':
    # Read data from files
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'trainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
    
    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, ""and %d unlabeled reviews\n" % (train["review"].size,test["review"].size, unlabeled_train["review"].size )

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #sentences=cPickle.load(open('sentences.p', 'rb'))
    #sentences = word2vec.Text8Corpus('data/alldata.txt')
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    # Set values for various parameters
    num_features = 200    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 16       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    #model = word2vec.Word2Vec(sentences, workers=num_workers,size=num_features, min_count = min_word_count,window = context, sample = downsampling, seed=1)
    #model.init_sims(replace=True)
    #model_name = "200features_10minwords_10context"
    #model.save(model_name)
    #model = word2vec.Word2Vec.load("200features_10minwords_10context")
    #
    '''
    f=open('data/alldata.txt','r')
    print('vectorizing... ')    
    vectorizer=TfidfVectorizer(min_df=2)
    X = vectorizer.fit_transform(f)
    idf_score = vectorizer._tfidf.idf_
    temp= vectorizer.get_feature_names()
    feature_names={}
    i=0
    for word in temp:
    		feature_names[word]=i
    		i=i+1
    '''
    print "Creating average feature vecs for training reviews"
    #trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features,feature_names,idf_score)
    #cPickle.dump(trainDataVecs, open('save_train_weighted.p', 'wb'))
    trainDataVecs = cPickle.load(open('save_train_weighted.p', 'rb'))
    
    print "Creating average feature vecs for test reviews"
    #testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features,feature_names,idf_score)
    #cPickle.dump(testDataVecs, open('save_test_weighted.p', 'wb'))
    testDataVecs = cPickle.load(open('save_test_weighted.p', 'rb'))
   
    X_train, y_train = load_svmlight_file("data/train.txt")
    X_test, y_test = load_svmlight_file("data/test.txt")
    #newTrain = np.hstack((trainDataVecs, X_train.toarray()))
    #newTest = np.hstack((testDataVecs, X_test.toarray()))
    newTrain=trainDataVecs
    newTest=testDataVecs
    ######################              LogisticRegression				####################
    #print "Fitting a L2 LogisticRegression classifier to labeled training data..."
    #for i in drange(0.1,10.0,0.3):
    #		clf=LogisticRegression(penalty='l1',C=i)
    #		clf.fit(newTrain, y_train)
    #		print i,"------------",clf.score(newTest,y_test)
    
    ######################              SVM				####################
    print "Fitting a Linear SVM classifier to labeled training data..."
    for i in drange(1100.0,2000.0,10.0):
    		clf = svm.SVC(kernel='rbf',C=i)
    		#clf=svm.LinearSVC(C=i)
    		clf.fit(newTrain, y_train)
    		print i,"------------",clf.score(newTest,y_test)
