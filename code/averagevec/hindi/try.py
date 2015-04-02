#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import pandas as pd
import chardet  
import os
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import word2vec
import cPickle
from sklearn import svm
from nltk import word_tokenize
from sklearn import cross_validation
from sklearn import naive_bayes
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk import word_tokenize
def tokenize(text):
	tokens = word_tokenize(text)
	stems = []
	for item in tokens:
		stems.append(item)
	return stems

np.set_printoptions(threshold='nan')
#with open('data/hindi_review_movie.txt') as f_in:
#	lines = list(line for line in (l.strip() for l in f_in) if line)
	
f=open('data/hindi_review_movie.txt')
corpus=[]
for line in f:
	line= line.strip()
	corpus.append(line)
decoded = [x.decode(chardet.detect(x)['encoding']) for x in (corpus)]
	
vectorizer = TfidfVectorizer(min_df=1,tokenizer=tokenize)
x=vectorizer.fit_transform(decoded).toarray()
y= vectorizer.get_feature_names()
#for word in y:
#	print word.encode('utf-8')
print x[0]
