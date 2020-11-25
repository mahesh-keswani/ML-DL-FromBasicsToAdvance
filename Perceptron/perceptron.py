import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime

def get_data():
	N = 500
	b = 0.1

	w = np.array([-0.5, 0.5])
	X = np.random.random((N, 2))*2 - 1

	Y = np.sign(X.dot(w) + b)
	return X, Y

class Perceptron(object):
	def fit(self, X, Y, learning_rate = 0.1, epochs = 1000, plot_cost = False):
		N, D = X.shape
		self.w = np.random.randn(D) / np.sqrt(D)
		self.b = 0

		N = len(Y)
		costs = []
		for epoch in range(epochs):
			Yhat = self.predict(X)
			# getting the indices of all the misclassified samples
			incorrect = np.nonzero(Y != Yhat)[0]
			if len(incorrect) == 0:
				break

			# taking the random index from misclassified samples
			random_index = np.random.choice(incorrect)
			# w = w + learning_rate * x * y
			self.w += learning_rate * X[random_index] * Y[random_index]
			self.b += learning_rate * Y[random_index]

			costs.append(len(incorrect) / N)

		if plot_cost:
			plt.plot(costs)
			plt.show()

	def predict(self, X):
		return np.sign(X.dot(self.w) + self.b)

	def score(self, X, Y):
		predictions = self.predict(X)
		return np.mean(predictions == Y)

if __name__ == '__main__':
	X, y = get_data()
	# print(X[:5])
	# print(y)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
	plt.show()

	Ntrain = len(y) // 2
	X_train, Y_train = X[:Ntrain], y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], y[Ntrain:]

	t0 = datetime.now()
	# no depth
	dt = Perceptron()
	dt.fit(X_train, Y_train, epochs = 100, plot_cost = True)

	print("Train accuracy: {}".format(dt.score(X_train, Y_train)))
	print("Test accuracy: {}".format(dt.score(Xtest, Ytest)))

	print("Took: {}".format(datetime.now() - t0))