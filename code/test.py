from gensim.models import word2vec

model = word2vec.Word2Vec.load_word2vec_format('amazon_mp3_review.bin', binary=True)
#model.most_similar(['girl', 'father'], ['boy'], topn=3)

print model.most_similar('good', topn=5)
#more_examples = ["he his she", "big bigger bad", "going went being"]
#for example in more_examples:
#	a, b, x = example.split()
#	predicted = model.most_similar([x, b], [a])[0][0]
#	print "'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted)
	
#print model.doesnt_match("breakfast cereal dinner lunch".split())
