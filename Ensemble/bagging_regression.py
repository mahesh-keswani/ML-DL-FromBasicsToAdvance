import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle

N = 100
X = np.linspace(0, 10, N).reshape(-1, 1)
Y = 2 * X[:, 0] + np.random.randn(N) * 0.5

train_size = int(0.9 * N)
Xtrain, Ytrain = X[:train_size], Y[:train_size]
Xtest, Ytest = X[train_size:], Y[train_size:]

model = DecisionTreeRegressor()
model.fit(Xtrain, Ytrain)
print("Score for 1 DecisionTreeRegressor", model.score(Xtest, Ytest))

plt.plot(Xtrain[:, 0], Ytrain, label = 'actual')
plt.plot(Xtrain[:, 0], model.predict(Xtrain), label = 'predictions from 1 model')
plt.legend()
plt.show()

class BaggedTreeRegressor:

	def __init__(self, B):
		self.B = B

	def fit(self, X, Y):
		self.models = []
		N = len(X)
		for b in range(self.B):
			# default sample with replacement is true
			idx = np.random.choice(N, size = N)
			Xb, Yb = X[idx], Y[idx]

			model = DecisionTreeRegressor()
			model.fit(Xb, Yb)
			self.models.append(model)

	def predict(self, X):
		N = len(X)
		predictions = np.zeros(N)
		for model in self.models:
			predictions += model.predict(X)

		print(X.shape, predictions.shape)
		return predictions / self.B

	def score(self, X, Y):
		# returning R^2
		numerator = Y - self.predict(X)
		denominator = Y - Y.mean()
		return 1 - numerator.dot(numerator) / denominator.dot(denominator)


model = BaggedTreeRegressor(200)
model.fit(Xtrain, Ytrain)

print("Score for bagged trees", model.score(Xtest, Ytest))

plt.plot(Xtrain[:, 0], Ytrain, label = 'actual')
plt.plot(Xtrain[:, 0], model.predict(Xtrain), label = 'predictions from bagged model')
plt.legend()
plt.show()


























