from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('data/hindi_review_movie.txt')

model = word2vec.Word2Vec(sentences, size=500,workers=8)
#model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
model.save('hindi_review_movie.model')
model.save_word2vec_format('hindi_review_movie.model.bin', binary=True)
model.save_word2vec_format('hindi_review_movie.model.txt', binary=False)


