import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from collections import Counter
import pickle
from sklearn.externals import joblib

class VoteClassifier(SklearnClassifier):

	def __init__(self,labels,classifiers):
		self.labels=labels
		self.classifiers=[SklearnClassifier(clf) for clf in classifiers]

	#Returns list of labels (target values)
	def labels():           
		return self.labels

	def train(self,featureSets):
		for clfIndex in range(len(self.classifiers)):
			self.classifiers[clfIndex].train(featureSets)
			print("Completed Training {}".format(clfIndex))

	def classify(self,featureSet):
		votes=[]
		for clf in self.classifiers:
			votes.append(clf.classify(featureSet))
		return self.mode(votes)

	#Required for classifying multiple featureSet during determination of accuracy
	def classify_many(self,featureSets):      
		return [self.classify(fs) for fs in featureSets]

	#Count the ratio of classifiers who agree with the majority predicted label
	def confidence(self,featureSet):          
		votes=[]
		for clf in self.classifiers:
			votes.append(clf.classify(featureSet))
		countMaj=votes.count(self.mode(votes))
		return (countMaj*1.0/len(votes))

	#Python 2.7 doesn't have a built-in statistics mode, so Counter is used
	def mode(self,votes):                    
		data=Counter(votes)
		return data.most_common(1)[0][0]
