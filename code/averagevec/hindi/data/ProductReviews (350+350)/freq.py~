#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import cPickle

f=open("Positive.txt",'r')
g=open("Negative.txt",'r')

dict_pos={}
dict_neg={}
word_list=[]
for line in f:
	line=line.rstrip()
	x=line.split(' ')
	for word in x:
		if word.decode('utf-8') in dict_pos:
			dict_pos[word.decode('utf-8')]=dict_pos[word.decode('utf-8')]+1
		else:
			dict_pos[word.decode('utf-8')]=1
			word_list.append(word.decode('utf-8'))
for line in g:
	line=line.rstrip()
	x=line.split(' ')
	for word in x:
		if word.decode('utf-8') in dict_neg:
			dict_neg[word.decode('utf-8')]=dict_neg[word.decode('utf-8')]+1
		else:
			dict_neg[word.decode('utf-8')]=1
			if word.decode('utf-8') not in word_list:
				word_list.append(word.decode('utf-8'))

for key in dict_pos:
	dict_pos[key]=float(dict_pos[key])/float(len(dict_pos))
	
for key in dict_neg:
	dict_neg[key]=float(dict_neg[key])/float(len(dict_neg))
	
score={}
for word in word_list:
	if word in dict_pos and word in dict_neg:
		score[word]=max(float(dict_pos[word])/float(dict_neg[word]),float(dict_neg[word])/float(dict_pos[word]))
	elif word in dict_pos:
		score[word]=float(dict_pos[word])
	else:
		score[word]=float(dict_neg[word])

cPickle.dump(score, open('score.p', 'wb'))
#cPickle.load(open('newTrain_worddoctfidf.p', 'rb'))
