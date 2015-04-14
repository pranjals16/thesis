from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('wikipedia_latest')

model = word2vec.Word2Vec(sentences, size=300,workers=8,window=10)
#model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
model.save('wikipedia_latest_context10')
model.save_word2vec_format('wikipedia_latest_context10.bin', binary=True)
model.save_word2vec_format('wikipedia_latest_context10.txt', binary=False)


# 4:33 PM collected 7875317 word types from a corpus of 3543978136 words and 3543979 sentences

#22:48 finished
