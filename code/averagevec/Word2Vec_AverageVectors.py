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



# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counterfrom sklearn import naive_bayes
from sklearn.ensemble import AdaBoostClassifier
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
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
    '''
    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from training set"
    for review in train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print "Parsing sentences from unlabeled set"
    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
    cPickle.dump(sentences, open('sentences.p', 'wb'))
    '''
    sentences=cPickle.load(open('sentences.p', 'rb'))
    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
    # Set values for various parameters
    num_features = 1000    # Word vector dimensionality
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
    model = word2vec.Word2Vec.load("300features_40minwords_10context")
    # ****** Create average vectors for the training and test sets
    #
    print "Creating average feature vecs for training reviews"
    trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features )
    #cPickle.dump(trainDataVecs, open('save_train.p', 'wb'))
    #trainDataVecs = cPickle.load(open('save_train.p', 'rb'))
    
    print "Creating average feature vecs for test reviews"
    testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features )
    #cPickle.dump(testDataVecs, open('save_test.p', 'wb'))
    #testDataVecs = cPickle.load(open('save_test.p', 'rb'))
    '''
    ######################              AdaBoostClassifier				####################
    print "Fitting a Ada Boost classifier to labeled training data..."
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(trainDataVecs, train["sentiment"])
    print clf.score(testDataVecs,test["sentiment"])
    #------------------------------------------------------------------------------------------
    ######################              Naive Bayes				####################
    print "Fitting a Naive Bayes classifier to labeled training data..."
    clf = naive_bayes.GaussianNB()
    clf.fit(trainDataVecs, train["sentiment"])
    print clf.score(testDataVecs,test["sentiment"])
    #------------------------------------------------------------------------------------------
    ######################              LogisticRegression				####################
    print "Fitting a LogisticRegression classifier to labeled training data..."
    clf = LogisticRegression(penalty='l1')
    clf.fit(trainDataVecs, train["sentiment"])
    print clf.score(testDataVecs,test["sentiment"])
    #------------------------------------------------------------------------------------------
    '''
    #------------------------------------------------------------------------------------------
    ######################              SVM				####################
    print "Fitting a SVM classifier to labeled training data..."
    clf = svm.LinearSVC(C=0.9)
    clf.fit(trainDataVecs, train["sentiment"])
    print clf.score(testDataVecs,test["sentiment"])
    
    #------------------------------------------------------------------------------------------
