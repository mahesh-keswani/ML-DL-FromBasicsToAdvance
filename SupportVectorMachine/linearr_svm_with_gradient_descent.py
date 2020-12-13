from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt 

# Note: source of plot_decision_boundary: github/lazyprogrammer/svm
def get_clouds():
	N = 1000
	c1 = np.array([2, 2])
	c2 = np.array([-2, -2])
	# c1 = np.array([0, 3])
	# c2 = np.array([0, 0])
	X1 = np.random.randn(N, 2) + c1
	X2 = np.random.randn(N, 2) + c2
	X = np.vstack((X1, X2))
	Y = np.array([-1]*N + [1]*N)
	return X, Y

class LinearSVM:
	def __init__(self, C=1.0):
		self.C = C

	def _objective(self, margins):
		return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1 - margins).sum()

	def fit(self, X, Y, lr = 1e-5, n_iters = 400):
		N, D = X.shape
		self.N = N
		self.w = np.random.randn(D)
		self.b = 0

		losses = []
		for _ in range(n_iters):
			margins = Y * self._decision_function(X)
			loss = self._objective(margins)
			losses.append(loss)

			# getting the indices for which prediction was not confident
			idx = np.where(margins < 1)[0]

			grad_w = self.w - self.C * Y[idx].dot(X[idx])
			self.w = self.w - lr * grad_w

			grad_b = -self.C * Y[idx].sum()
			self.b = self.b - lr * grad_b

		self.support = np.where( (Y * self._decision_function(X)) <= 1)[0]
		print("Num of support", len(self.support))

		print("Weights", self.w)
		print("Bias", self.b)

		plt.plot(losses)
		plt.title("Loss per iteration")
		plt.show()

	def _decision_function(self, X):
		return X.dot(self.w) + self.b

	def predict(self, X):
		return np.sign(self._decision_function(X))

	def score(self, X, Y):
		P = self.predict(X)
		return np.mean(P == Y)

def clouds():
	X, Y = get_clouds()
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3)
	#                                    lr,   n_iters 
	return Xtrain, Xtest, Ytrain, Ytest, 1e-3, 200

def breast_cancer_data():
	data = load_breast_cancer()
	X, Y = data.data, data.target
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3)
	print(Xtrain.shape)
	#                                    lr,   n_iters 
	return Xtrain, Xtest, Ytrain, Ytest, 1e-3, 200


if __name__ == '__main__':
	Xtrain, Xtest, Ytrain, Ytest, lr, n_iters = clouds()

	sc = StandardScaler()
	Xtrain = sc.fit_transform(Xtrain)
	Xtest = sc.transform(Xtest)

	plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain, s=100)
	plt.show()

	model = LinearSVM(C = 1.0)
	t0 = datetime.now()
	model.fit(Xtrain, Ytrain, lr=lr, n_iters=n_iters)
	print("Score: ", model.score(Xtest, Ytest))
	print("Took", datetime.now() - t0)

	Xtrain, Xtest, Ytrain, Ytest, lr, n_iters = breast_cancer_data()

	sc = StandardScaler()
	Xtrain = sc.fit_transform(Xtrain)
	Xtest = sc.transform(Xtest)

	model = LinearSVM(C = 1.0)

	t0 = datetime.now()
	model.fit(Xtrain, Ytrain, lr=lr, n_iters=n_iters)
	print("Score: ", model.score(Xtest, Ytest))
	print("Took", datetime.now() - t0)

