import nltk
import os
import pickle
import voteclf
import sys
from django.shortcuts import render
from django.http import HttpResponse
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#The pickled object in the data/classifier.pickle has the module attribute set to __main__
# as can be seen as:-
							# ccopy_reg
							# _reconstructor
							# p0
							# (c__main__                 <======== name of module
							# VoteClassifier             <======== name of class
							# p1
							# c__builtin__
							# object
							# p2
# but the class definition lies in the voteclf module (voteclf.py file)
#thus for successful unpickling of data, we need to set the sys.modules['__main__'] key to the voteclf module imported in this file
import __main__
sys.modules['__main__']=voteclf

def index(request):
	return render(request, 'sentiment/home.html')

def sentiment(request):
	try:
		sentence=request.POST['feed']

		words=preprocess(sentence)

		classifier=getPickled('data/classifier.pickle')
		mostCommon=getPickled('data/words.pickle')

		featureSet={}
		for w in mostCommon:
			featureSet[w]=(w in words)

		sentiment=str(classifier.classify(featureSet))
		confidence=100*round(classifier.confidence(featureSet),4)

		return render(request, 'sentiment/result.html', {'sentence': sentence, 'sentiment': sentiment, 'confidence': confidence})

	except KeyError as e:
		return HttpResponse('<h1>{}</h1>'.format('Post Data not Recieved'))

def getPickled(fileName):
	mod = os.path.dirname(__file__)  # get current directory
	fPath = os.path.join(mod,fileName)
	with open(fPath, 'rb') as f:
		return pickle.load(f)

def preprocess(sentence):
	stopWordList=stopwords.words("english")
	remPunc=nltk.RegexpTokenizer(r'[a-zA-Z]+')
	words=remPunc.tokenize(sentence)
	words=[w.lower() for w in words]
	words=[w for w in words if not w in stopWordList]
	ps=PorterStemmer()
	words=[ps.stem(w) for w in words]
	return words


