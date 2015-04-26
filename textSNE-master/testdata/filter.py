f=open("../../code/wikipedia_latest_context5.txt","r")
i=0
for line in f:
	if(i<=5000):
		x=line.split()
		#print line.strip()
		#print x[0]
		for z in range(1,len(x)):
			print x[z],
		print
	else:
		break
	i=i+1
	
'''
words=['central','country','district', 'village','independent','organization','movement','group','groups','episode','series','show','shows','television','radio','studies','research','field','course','summer','plant','species','air','water','natural', 'good','better','happy','sweet','love','vein','blood','nerve','heart','lungs','stomach','my','his','her','I']
for line in f:
	x=line.split()
	if(x[0] in words):
		#print x[0]
		for i in range(1,len(x)):
			print x[i],
		print
		i=i+1
'''
