#!/usr/bin/env python
# -*- coding: utf-8 -*- 
f=open("new_movie_reviews2.txt","r")

for line in f:
	sent=line.lstrip().replace('?','।').replace('!','।').split('।')
	for x in sent:
		if x.rstrip():
			print x
