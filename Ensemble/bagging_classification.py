import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from util import plot_decision_boundary

N = 500
D = 2

X = np.random.randn(N, D)
seperator = 2

X[:125] += np.array([seperator, seperator])
X[125:250] += np.array([seperator, -seperator])
X[250:375] += np.array([-seperator, seperator])
X[375:] += np.array([-seperator, -seperator])

Y = np.array([1]*125 + [0] * 250 + [1] * 125)

train_size = int(0.9 * N)
Xtrain, Ytrain = X[:train_size], Y[:train_size]
Xtest, Ytest = X[train_size:], Y[train_size:]

plt.scatter(X[:, 0], X[:, 1], c=Y, s=100)
plt.show()

model = DecisionTreeClassifier()
model.fit(Xtrain, Ytrain)
print("Score for 1 model", model.score(Xtest, Ytest))

plt.scatter(X[:, 0], X[:, 1], c=Y, s=100)
plot_decision_boundary(X, model)
plt.show()

class BaggedTreeClassifier:
	def __init__(self, B):
		self.B = B

	def fit(self, X, Y):
		N = len(X)
		self.models = []
		for b in range(self.B):

			idx = np.random.choice(N, size = N)
			Xb, Yb = X[idx], Y[idx]

			model = DecisionTreeClassifier(max_depth = 2)
			model.fit(Xb, Yb)
			self.models.append(model)

	def predict(self, X):
		predictions = np.zeros(len(X))
		for model in self.models:
			predictions += model.predict(X)

		return np.round(predictions / self.B)

	def score(self, X, Y):
		P = self.predict(X)

		return np.mean(Y == P )

model = BaggedTreeClassifier(200)
model.fit(Xtrain, Ytrain)
print("Score for BaggedTreeClassifier", model.score(Xtest, Ytest))

plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), s=100)
plot_decision_boundary(X, model)
plt.show()




































