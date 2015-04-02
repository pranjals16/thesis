#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from gensim.models import word2vec

model = word2vec.Word2Vec.load_word2vec_format('hindi.model.bin', binary=True)
#print model.similarity('भारत'.decode('utf8'), 'चीन'.decode('utf8'))
#print model['कामयाब'.decode('utf8')]
#print model
#x= model.most_similar_cosmul(['चीन'.decode('utf8'), 'मुम्बई'.decode('utf8')], ['भारत'.decode('utf8')], topn=10)
x= model.most_similar_cosmul(['रसायन'.decode('utf8')], topn=5)
for word in x:
	for s in word:
		print s


#more_examples = ["he his she", "big bigger bad", "going went being"]
#for example in more_examples:
#	a, b, x = example.split()
#	predicted = model.most_similar([x, b], [a])[0][0]
#	print "'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted)

#w1='नेता'.decode('utf8')
#w2='मंत्री'.decode('utf8') 
#w3='सरकार'.decode('utf8') 
#w4='उद्योग'.decode('utf8')
#print model.doesnt_match([w1,w2,w3,w4])

