f=open("../temp_wiki_10_filter","r")

for line in f:
	x=line.rstrip().split(' ')
	print "\\node[] at ("+x[1]+","+x[2]+") {\\sizeThree \\bf\\serifbb "+x[0]+"};"
'''
for line in f:
	x=line.split(' ')
	for i in range(1,200):
		print x[i],
	print
'''
