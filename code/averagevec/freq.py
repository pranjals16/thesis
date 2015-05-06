#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import cPickle
import math
from sklearn.feature_extraction.text import CountVectorizer
f=open("data/train-pos.txt",'r')
g=open("data/train-neg.txt",'r')

dict_pos={}
dict_neg={}
word_list=[]
for line in f:
	line=line.strip().rstrip()
	x=line.split(' ')
	for word in x:
		if word in dict_pos:
			dict_pos[word]=dict_pos[word]+1
		else:
			dict_pos[word]=1
			word_list.append(word)

print "Boundary 1",len(dict_pos)
for line in g:
	line=line.strip().rstrip()
	x=line.split(' ')
	for word in x:
		if word in dict_neg:
			dict_neg[word]=dict_neg[word]+1
		else:
			dict_neg[word]=1
			if word not in word_list:
				word_list.append(word)
print "Boundary 2",len(dict_neg)
score={}
for word in word_list:
	if word in dict_pos and word in dict_neg:
		score[word]=math.log(float(dict_pos[word])/float(dict_neg[word]),2)
	else:
		score[word]=1.0
cPickle.dump(score, open('score.p', 'wb'))
'''

for key in dict_pos:
	dict_pos[key]=float(dict_pos[key])/float(len(dict_pos))
	
for key in dict_neg:
	dict_neg[key]=float(dict_neg[key])/float(len(dict_neg))
print "Boundary 3"
score={}
for word in word_list:
	if word in dict_pos and word in dict_neg:
		score[word]=max(float(dict_pos[word])/float(dict_neg[word]),float(dict_neg[word])/float(dict_pos[word]))*10
	elif word in dict_pos:
		score[word]=float(dict_pos[word])*10
	else:
		score[word]=float(dict_neg[word])*10

cPickle.dump(score, open('score.p', 'wb'))
#cPickle.load(open('newTrain_worddoctfidf.p', 'rb'))
'''
