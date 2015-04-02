f=open("english.model.txt","r")
i=0
for line in f:
	if(i<=1000):
		print line.strip()
	else:
		break
	i=i+1
