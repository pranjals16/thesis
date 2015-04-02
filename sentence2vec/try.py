from gensim.models import word2vec, doc2vec
from word2vec import Sent2Vec, LineSentence
import logging
import sys
import os
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

#sentences = word2vec.Text8Corpus('english')
sentences = doc2vec.LabeledLineSentence('english')
model = doc2vec.Doc2Vec(sentences,workers=8,size=500,min_count=5,window=8)
#model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
model.save('amazon_mp3_review.model')
print model.most_similar("SENT_0")
