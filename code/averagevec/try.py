import pandas as pd
import os
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn import svm,grid_search
from sklearn.ensemble import RandomForestClassifier

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

X_train, y_train = load_svmlight_file("data/train.txt")
X_test, y_test = load_svmlight_file("data/test.txt")
'''
######################              LogisticRegression				####################
print "Fitting a LogisticRegression classifier to labeled training data..."
clf = LogisticRegression(penalty='l1')
clf.fit(X_train, y_train)
print clf.score(X_test,y_test)
#------------------------------------------------------------------------------------------
'''
######################              SVM				####################
print "Fitting a SVM classifier to labeled training data..."
parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 10.0]}
for i in drange(3.6,4.6,0.1):
	clf = svm.SVC(kernel='rbf',C=i)
	#clf = grid_search.GridSearchCV(svr, parameters)
	#clf = svm.LinearSVC(C=0.9)
	clf.fit(X_train, y_train)
	print i,"-------",clf.score(X_test,y_test)

#------------------------------------------------------------------------------------------
