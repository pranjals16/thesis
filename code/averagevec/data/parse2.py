#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ****** Read the two training sets and the test set
#
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
from KaggleWord2VecUtility2 import KaggleWord2VecUtility
from sklearn import svm
from nltk import word_tokenize

	
if __name__ == '__main__':
	train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    	test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    	unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__),  "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
	
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	for review in train["review"]:
		KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

	for review in unlabeled_train["review"]:
		KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
	for review in test["review"]:
		KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
