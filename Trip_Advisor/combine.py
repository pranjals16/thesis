f=open('out2','r')
g=open('out','r')

content=[]
rating=[]
for l in f:
	l=l.strip()
	content.append(l)
	
for l in g:
	if float(l)>=3.0:
		rating.append(1)
	else:
		rating.append(0)
print "sentiment"+"\t"+"review"
for i in range(0,len(rating)):
	print str(rating[i])+"\t"+content[i]
