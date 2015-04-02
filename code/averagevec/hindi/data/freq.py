#!/usr/bin/env python
# -*- coding: utf-8 -*- 
f=open('hindi_review_movie.txt','r')

counter=0
for line in f:
	word=line.rstrip().split(' ')
	if u'सिद्धार्थ'.encode('utf-8') in word:
		counter=counter+1
		continue
print counter
