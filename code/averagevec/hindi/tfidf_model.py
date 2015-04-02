#!/usr/bin/env python
# ****** Read the two training sets and the test set
#
import pandas as pd
import chardet  
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
def tokenize(text):
	tokens = word_tokenize(text)
	stems = []
	for item in tokens:
		stems.append(item)
	return stems
	
def makeFeatureVec(words, model,num_features,index2word_set,X,counter):
	# Function to average all of the word vectors in a given
	# paragraph
	#
	# Pre-initialize an empty numpy array (for speed)
	new_feature=X.shape[1]
	featureVec = np.zeros((num_features+new_feature,),dtype="float32")
	#
	nwords = 0.
	#
	for word in words:
		if word in index2word_set:
			temp = np.zeros((num_features+new_feature,),dtype="float32")
			temp[0:500]=model[word]
			temp[500:]=X[counter]
			#print X[counter][word]
			featureVec = np.add(featureVec,temp)
			nwords = nwords + 1.
	#
	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)
	#print len(index2word_set2)
	return featureVec


def getAvgFeatureVecs(reviews, model, num_features,X):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    new_feature=X.shape[1]
    reviewFeatureVecs = np.zeros((len(reviews),num_features+new_feature),dtype="float32")
    index2word_set = set(model.index2word)
    
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       #if counter%1000. == 0.:
       print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features,index2word_set,X,counter)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
	clean_reviews = []
	for review in reviews["review"]:
		raw=review.decode('utf-8')
		tokens = word_tokenize(raw)
		#print tokens[:10]
		#words = [w for w in words]
		clean_reviews.append(tokens)
	return clean_reviews



if __name__ == '__main__':

    # Read data from files
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'iitb.tsv'), header=0, delimiter="\t", quoting=3 )
    
    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews\n" % (train["review"].size)

    # Set values for various parameters
    num_features = 500    # Word vector dimensionality
    min_word_count = 1   # Minimum word count
    num_workers = 8       # Number of threads to run in parallel
    context = 5          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    # Initialize and train the model (this will take some time)
    sentences = word2vec.Text8Corpus('data/hindi_review_movie.txt')
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
    # tf-idf
    f=open('data/hindi_review_movie.txt')
    corpus=[]
    for line in f:
		line= line.strip()
		corpus.append(line)
    decoded = [x.decode(chardet.detect(x)['encoding']) for x in (corpus)]

    vectorizer = TfidfVectorizer(min_df=1,tokenizer=tokenize)
    X=vectorizer.fit_transform(decoded).toarray()
    

    print "Creating average feature vecs for training reviews"
    trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features,X)
    #cPickle.dump(trainDataVecs, open('save_train_mix.p', 'wb'))
    #trainDataVecs = cPickle.load(open('save_train_mix.p', 'rb'))

    ######################              SVM				####################
    print "Fitting a SVM classifier to labeled training data..."
    clf = svm.LinearSVC()
    clf.fit(trainDataVecs, train["sentiment"])
    scores= cross_validation.cross_val_score(clf, trainDataVecs,train["sentiment"], cv=50)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
