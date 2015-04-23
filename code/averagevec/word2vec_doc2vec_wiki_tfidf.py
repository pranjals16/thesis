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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def makeTfidfFeatures(traindata, testdata):
    print('vectorizing... ')    
    tfv = TfidfVectorizer(min_df=2,  max_features=15000,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')
    X_all = traindata + testdata
    lentrain = len(traindata)
    tfv.fit(X_all)
    X_all = tfv.transform(X_all)
    trainfeatures = X_all[:lentrain]
    testfeatures = X_all[lentrain:]
    return trainfeatures, testfeatures
    
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
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
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
    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    #sentences=cPickle.load(open('sentences.p', 'rb'))
    sentences = word2vec.Text8Corpus('data/alldata.txt')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
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
    '''
    traindata = []
    for i in range( 0, len(train["review"])):
    		traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], False)))
    testdata = []
    for i in range(0,len(test["review"])):
    		testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], False)))
    cPickle.dump(traindata, open('traindata.p', 'wb'))
    cPickle.dump(testdata, open('testdata.p', 'wb'))
    '''
    #traindata=cPickle.load(open('traindata.p', 'rb'))
    #testdata=cPickle.load(open('testdata.p', 'rb'))
    #trainTfidfData, testTfidfData = makeTfidfFeatures(traindata, testdata)
    #cPickle.dump(trainTfidfData, open('tfidf_train.p', 'wb'))
    #cPickle.dump(testTfidfData, open ('tfidf_test.p', 'wb'))
    trainTfidfData = cPickle.load(open('tfidf_train.p', 'rb'))
    testTfidfData = cPickle.load(open('tfidf_test.p', 'rb'))
    print trainTfidfData.shape,testTfidfData.shape
    print "Creating average feature vecs for training reviews"
    #trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features )
    #cPickle.dump(trainDataVecs, open('save_train.p', 'wb'))
    trainDataVecs = cPickle.load(open('save_train.p', 'rb'))
    trainDataVecs2 = cPickle.load(open('save_train_mix_context10.p', 'rb'))
    
    print "Creating average feature vecs for test reviews"
    #testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features )
    #cPickle.dump(testDataVecs, open('save_test.p', 'wb'))
    testDataVecs = cPickle.load(open('save_test.p', 'rb'))
    testDataVecs2 = cPickle.load(open('save_test_mix_context10.p', 'rb'))
   
    X_train, y_train = load_svmlight_file("data/train.txt")
    X_test, y_test = load_svmlight_file("data/test.txt")
    newTrain3 = np.hstack((trainDataVecs, X_train.toarray()))
    newTest3 = np.hstack((testDataVecs, X_test.toarray()))
    newTrain2 = np.hstack((trainDataVecs2, newTrain3))
    newTest2 = np.hstack((testDataVecs2, newTest3))
    newTrain=np.hstack((newTrain2, trainTfidfData.toarray()))
    print "1st Loaded!!!"
    newTest=np.hstack((newTest2, testTfidfData.toarray()))
    print "2nd Loaded!!!"
    #cPickle.dump(newTrain, open('newTrain_worddoctfidf.p', 'wb'))
    #cPickle.dump(newTest, open('newTest_worddoctfidf.p', 'wb'))
    #newTrain=cPickle.load(open('newTrain_worddoctfidf.p', 'rb'))
    #newTest=cPickle.load(open('newTest_worddoctfidf.p', 'rb'))
    import gc
    gc.collect()
    ######################              LogisticRegression				####################
    print "Fitting a LogisticRegression classifier to labeled training data..."
    clf=LogisticRegression(penalty='l1')
    clf.fit(newTrain, y_train)
    print clf.score(newTest,y_test)
    #------------------------------------------------------------------------------------------
    ######################              SVM				####################
    print "Fitting a SVM classifier to labeled training data..."
    for i in drange(0.2,0.4,0.01):
    		#clf = svm.SVC(kernel='linear',C=i)
    		clf=svm.LinearSVC(C=i)
    		clf.fit(newTrain, y_train)
    		print i,"------------",clf.score(newTest,y_test)
    #------------------------------------------------------------------------------------------
