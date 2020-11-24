from get_data import get_mnist
import numpy as np 
import matplotlib.pyplot as plt
'''
	Note: We could have used SortedDictionaries but key will be the distance
	and if two distances are very close then, one will be overwritten
''' 
from sortedcontainers import SortedList
from datetime import datetime

class KNN(object):
	def __init__(self, k):
		self.k = k

	def fit(self, X, Y):
		self.X = X
		self.Y = Y

	def predict(self, X):
		# for predictions
		y = np.zeros(len(X))
		for i, x in enumerate(X):
			s = SortedList()
			# Note: In outer loop we are looping through X passed to predict
			# In inner loop we are parsing through self.X
			for j, xt in enumerate(self.X):
				diff = x - xt
				# using sum of squared differences
				distance = diff.dot(diff)
				# if len(SortedList) is less than k then simply add
				# no need to check anything
				if len(s) < self.k:
					# (distance, class)
					s.add((distance, self.Y[j]))
				else:
					if distance < s[-1][0]:
						del s[-1]
						s.add((distance, self.Y[j]))

			# Now creating dictionary of pattern 
			# {class1: count_1, class2: count2, ...}
			votes = {}
			for _, vote in s:
				votes[vote] = votes.get(vote, 0) + 1

			# Now sorting the votes based on maximum count
			max_votes = 0
			max_votes_class = -1
			for v, count in votes.items():
				if count > max_votes:
					max_votes = count
					max_votes_class = v 

			y[i] = max_votes_class

		return y 

	def score(self, X, Y):
		prediction = self.predict(X)
		return np.mean(prediction == Y)

if __name__ == '__main__':
	X, Y = get_mnist(limit = 2000)

	Ntrain = 1000
	Xtrain, Ytrain = X[:-Ntrain], Y[:-Ntrain]
	Xtest, Ytest = X[-Ntrain:], Y[-Ntrain:]

	for k in (1, 2, 3, 4, 5):
		knn = KNN(k)

		t0 = datetime.now()
		knn.fit(Xtrain, Ytrain)
		print("For K = {}, train score = {}".format(k, knn.score(Xtrain, Ytrain)))
		print("For K = {}, test score = {}".format(k, knn.score(Xtest, Ytest)))
		print("Took: {}".format(datetime.now() - t0))
		print("\n\n")

