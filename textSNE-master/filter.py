words=['electricity','supplies','equipment','fuel','industrial','agricultural','storage','custom','machine','million','billion','revenue','employees','society','societies','scholarship','courses','college',
'masters','bachelor','electoral','constituencies','seats','election','id','facebook','coibot','url','html','wikiproject','wikipedia','declined','rejected','approval','changed','riding','fishing','drivers',
'driven','driver','high','higher','highest','helpful','useful','successful','powerful','me','my','am','his','her']

f=open('temp_wiki_5','r')
for line in f:
	x=line.split()
	if(x[0] in words):
		print line.strip()
