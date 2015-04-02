from gensim.models import word2vec, doc2vec
from gensim import models
import logging,sys

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))
sentences = doc2vec.LabeledLineSentence('hindi2')
model = doc2vec.Doc2Vec(sentences, size=100, window=8, min_count=5, workers=8)
#model.save("hindi_doc2vec")
#model=doc2vec.Doc2Vec.load("hindi_doc2vec")
print model["SENT_0"]
