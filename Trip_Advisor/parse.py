import glob

read_files = glob.glob("Review_Texts/*")

for filename in read_files:
	f = open(filename, 'r') 
	entry = {} 
	for l in f: 
		l = l.strip() 
		colonPos = l.find('<Content>')
		if colonPos == -1: 
			if (entry):
				print entry['']
			entry = {} 
			continue 
		eName = l[:colonPos] 
		rest = l[colonPos+9:] 
		entry[eName] = rest
	if (entry):
		print entry['<Content>']
