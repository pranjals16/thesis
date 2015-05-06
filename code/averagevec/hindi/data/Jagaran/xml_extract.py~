import codecs,re
import string, operator
from xml.dom import minidom

def cleanup(data):
	data = data.strip()
	data = re.sub(ur'[\u0964|\u0965]|[\u0964-\u0970]|[0-9]', ur' ', data) # Hindi full stops | Hindi numbers | Eng numbers
	data = ''.join(ch for ch in data if ch not in string.punctuation) # Remove Puncts
	#data = ' '.join(w for w in data.split() if len(w)>1) # Remove single length word
	return data.strip()
	
xmldoc = minidom.parse('jagran.xml')
itemlist = xmldoc.getElementsByTagName('line') 

posReviews, negReviews = [], []
for node in itemlist:
	if node.childNodes[0].nodeType == node.TEXT_NODE:
		if node.attributes['sentiment'].value == 'pos':	posReviews += [cleanup(line) for line in node.childNodes[0].data.split('\n') if line.strip() != '']
		if node.attributes['sentiment'].value == 'neg':	negReviews += [cleanup(line) for line in node.childNodes[0].data.split('\n') if line.strip() != '']

posStr, negStr = '', ''
for r in posReviews:	posStr += r +'\n'
for r in negReviews:	negStr += r +'\n'
codecs.open('pos.txt','w','utf8').write(posStr)
codecs.open('neg.txt','w','utf8').write(negStr)

print len(posReviews), len(negReviews)

#for reviews in negReviews:
#	print reviews.encode('utf-8')
