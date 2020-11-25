import numpy as np 
import matplotlib.pyplot as plt 
from get_data import get_mnist
from datetime import datetime
'''
	Assumptions: Only binary classifier
	Node can have 0 or 2 children
'''
def entropy(y):
	N = len(y)
	s1 = (y == 1).sum()
	if s1 == 0 or s1 == N:
		return 0
	# probability of 1
	p1 = float(s1) / N
	p0 =  1 - p1

	return -p0*np.log2(p0) - p1*np.log2(p1)

class TreeNode(object):
	def __init__(self, depth = 0, max_depth = None):
		self.depth = depth
		self.max_depth = max_depth

	def fit(self, X, Y):
		# if there is only one sample OR Y contains only one value
		if len(Y) == 1 or len(set(Y)) == 1:
			self.col = None
			self.split = None
			self.left = None
			self.right = None
			self.prediction = Y[0]
		else:
			N, D = X.shape
			cols = range(D)
			# max information gain(ig)
			max_ig = 0
			best_col = None
			best_split = None
			# finding best column for splitting
			for col in cols:
				ig, split = self.find_split(X, Y, col)
				if ig > max_ig:
					max_ig = ig
					best_split = split
					best_col = col

			if max_ig == 0:
				self.left = None
				self.right = None
				self.col = None
				self.split = None
				self.prediction = np.round(Y.mean())
			else:
				self.col = best_col
				self.split = best_split

				# if we have reached max_depth, i.e time to make leaf
				if self.depth == self.max_depth:
					self.left = None
					self.right = None
					self.prediction = [
						np.round(Y[X[:, best_col] < best_split].mean()),
						np.round(Y[X[:, best_col] >= best_split].mean())
					]
				else:
					# indices for the left child node
					left_idx = (X[:, best_col] < best_split)
					X_left = X[left_idx]
					Yleft = Y[left_idx]

					self.left = TreeNode(self.depth + 1, self.max_depth)
					self.left.fit(X_left, Yleft)

					# indices for the right child node
					right_idx = (X[:, best_col] >= best_split)
					X_right = X[right_idx]
					Yright = Y[right_idx]

					self.right = TreeNode(self.depth + 1, self.max_depth)
					self.right.fit(X_right, Yright)

	def find_split(self, X, Y, col):
		x_values = X[:, col]
		# argsort() returns the indexes of X in sorted order
		sort_idx = np.argsort(x_values)
		x_values = x_values[sort_idx]
		y_values = Y[sort_idx]

		boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
		best_split = None
		max_ig = 0
		for i in boundaries:
			split = (x_values[i] + x_values[i + 1]) / 2
			ig = self.information_gain(x_values, y_values, split)
			if ig > max_ig:
				max_ig = ig
				best_split = split

		return max_ig, best_split

	def information_gain(self, x, y, split):
		y0 = y[x < split]
		y1 = y[x >= split]
		N = len(y)
		y0len = len(y0)

		if y0len == 0 or y0len == N:
			return 0
		p0 = float(len(y0)) / N
		p1 = 1 - p0
		return entropy(y) - p0*entropy(y0) - p1*entropy(y1)


	def predict_one(self, x):
		if self.col is not None and self.split is not None:
			feature = x[self.col]
			if feature < self.split:
				if self.left:
					# we have not reached the leaf yet
					p = self.left.predict_one(x)
				else:
					# we have reached the leaf in left direction
					p = self.prediction[0]
			else:
				if self.right:
					p = self.right.predict_one(x)
				else:
					# we have reached the leaf in left direction
					p = self.prediction[1]
		else:
			# # when there is only one sample OR there is only one label OR when information gain = 0
			p = self.prediction

		return p 

	def predict(self, X):
		N = len(X)
		P = np.zeros(N)
		for i in range(N):
			P[i] = self.predict_one(X[i])
		return P 

class DecisionTree:

	def __init__(self, max_depth = None):
		self.max_depth = max_depth

	def fit(self, X, Y):
		self.root = TreeNode(max_depth = self.max_depth)
		self.root.fit(X, Y)

	# def predict(self, X):
	# 	self.root.predict(X)

	def score(self, X, Y):
		P = self.root.predict(X)
		return np.mean(P == Y)


if __name__ == '__main__':
	X, Y = get_mnist(limit = 5000)
	# getting all the indices where y == 0 or y == 1
	idx = np.logical_or(Y == 0, Y == 1)
	X = X[idx]
	Y = Y[idx]

	Ntrain = len(Y) // 2
	X_train, Y_train = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	t0 = datetime.now()
	# no depth
	dt = DecisionTree(max_depth = 3)
	dt.fit(X_train, Y_train)

	print("Train accuracy: {}".format(dt.score(X_train, Y_train)))
	print("Test accuracy: {}".format(dt.score(Xtest, Ytest)))

	print("Took: {}".format(datetime.now() - t0))