#!/usr/bin/env python
import pandas as pd
import logging
import numpy as np
from gensim.models import word2vec
import cPickle
from KaggleWord2VecUtility import KaggleWord2VecUtility

num_features = 300 
model = word2vec.Word2Vec.load("300features_40minwords_10context")
model_full = word2vec.Word2Vec.load("../wikipedia_latest.model")
nwords = 0.
index2word_set = set(model.index2word)
#index2word_set2 = set(model_full.index2word)

featureVec = np.zeros((num_features+200,),dtype="float32")
temp = np.zeros((num_features+200,),dtype="float32")
temp[200:500]=model["good"]
temp[0:200]=model_full["good"]
featureVec = np.add(featureVec,temp)
