#!/usr/bin/env python
import pandas as pd
import os
import logging
import numpy as np  # Make sure that numpy is imported
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import cPickle
from sklearn import svm,grid_search
from sklearn import naive_bayes
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
from EnsembleClassifier import EnsembleClassifier
from sklearn.linear_model import LinearRegression

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step
		
if __name__ == '__main__':
	print "Creating average feature vecs for training reviews"
	trainDataVecs = cPickle.load(open('save_train.p', 'rb'))

	print "Creating average feature vecs for test reviews"
	testDataVecs = cPickle.load(open('save_test.p', 'rb'))
	trainTfidfData = cPickle.load(open('tfidf_train.p', 'rb'))
	testTfidfData = cPickle.load(open('tfidf_test.p', 'rb'))
	X_train, y_train = load_svmlight_file("data/train.txt")
	X_test, y_test = load_svmlight_file("data/test.txt")
	newTrain2 = np.hstack((trainDataVecs, X_train.toarray()))
	newTest2 = np.hstack((testDataVecs, X_test.toarray()))
	newTrain=np.hstack((newTrain2, trainTfidfData.toarray()))
	print "1st Loaded!!!"
	newTest=np.hstack((newTest2, testTfidfData.toarray()))
	print "2nd Loaded!!!"
	print newTrain.shape,newTest.shape
	#newTest=testDataVecs
	clf1 = svm.LinearSVC(C=0.33)
	#clf2 = RandomForestClassifier(n_estimators=20)
	clf3 = LogisticRegression(penalty='l2',C=4.3)
	#clf4 = GaussianNB()
	clf5 =AdaBoostClassifier(n_estimators=100)
	np.random.seed(123)
	eclf = EnsembleClassifier(clfs=[clf1, clf3, clf5], voting='hard')
	
	for clf, label in zip([clf1, clf3, clf5, eclf], ['Linear SVM', 'LogisticRegression', 'AdaBoost', 'Ensemble']):
		clf.fit(newTrain, y_train)
		print clf.score(newTest,y_test),"---",label
