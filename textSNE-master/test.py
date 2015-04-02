# -*- coding: utf-8 -*-
#!/usr/bin/env python
import string, numpy, gzip
#o = gzip.open("testdata/english-embeddings.turian.txt.gz", "rb")
o = gzip.open("testdata/hindi.gz", "rb")
titles, x = [], []
for l in o:
    toks = string.split(l)
    titles.append(toks[0])
    x.append([float(f) for f in toks[1:]])
x = numpy.array(x)

#from tsne import tsne
from calc_tsne import tsne
#out = tsne(x, no_dims=2, perplexity=30, initial_dims=30, USE_PCA=False)
#out = tsne(x, no_dims=2, perplexity=30, initial_dims=30, use_pca=False)
out = tsne(x, no_dims=2, perplexity=100, initial_dims=30)

words=set(line.strip() for line in open('testdata/hindi_icon_words.txt'))

import render
points=[]
for title, point in zip(titles, out):
	if title in words:
		points.append([title,point[0], point[1]])
		print title
		
render.render(points, "test-output.rendered.jpg", width=640, height=480)
#import render
#render.render([(title, point[0], point[1]) for title, point in zip(titles, out)], "test-output.rendered.png", width=3000, height=1800)
