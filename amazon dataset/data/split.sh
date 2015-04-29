FILENAME='train-neg.txt'
gawk 'BEGIN {srand()} {f = FILENAME (rand() <= 0.8 ? ".80" : ".20"); print > f}' neg.txt
#wc -l 100.txt*
