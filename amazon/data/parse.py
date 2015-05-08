import gzip
filename="amazon_mp3.tar.gz"
f = gzip.open(filename, 'r') 
entry = {} 
for l in f: 
	l = l.strip() 
	colonPos = l.find(':')
	if colonPos == -1: 
		if (entry and float(entry['[rating]'])<3.0):
			print entry['[fullText]']
		entry = {} 
		continue 
	eName = l[:colonPos] 
	rest = l[colonPos+1:] 
	entry[eName] = rest
if (entry and float(entry['[rating]'])<3.0):
	print entry['[fullText]']
