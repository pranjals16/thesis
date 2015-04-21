import pandas as pd
import os

train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__),  "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )

for review in test["review"]:
	print review
	
