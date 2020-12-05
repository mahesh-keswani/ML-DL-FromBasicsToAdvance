'''
	implementing random forest from scratch but selecting d features only once :|.
	Also comparing with bagging classifier 
'''
from util import get_xor
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score 

Xtrain, Ytrain, Xtest, Ytest = get_xor()

class NotAsRandomForest:

	def __init__(self, n_estimators):
		self.B = n_estimators

	def fit(self, X, Y):
		N, D = X.shape
		M = int(np.sqrt(D))

		self.models = []
		self.features = []

		for b in range(self.B):
			model = DecisionTreeClassifier()

			# sample features
			feature_sample = np.random.choice(D, size = M, replace = False)

			# bootstrapping
			idx = np.random.choice(N, size = N, replace = True)
			Xb, Yb = X[idx], Y[idx]

			model.fit(Xb[:, feature_sample], Yb)

			self.features.append(feature_sample)
			self.models.append(model)

	def predict(self, X):
		predictions = np.zeros(len(X))
		for features, model in zip(self.features, self.models):
			predictions += model.predict(X[:, features])

		return np.round(predictions / self.B)

	def score(self, X, Y):
		P = self.predict(X)
		return np.mean(P == Y)

N_TREES = 100
our_model_scores = np.zeros(N_TREES)
bagging_scores = np.zeros(N_TREES)

for t in range(1, N_TREES):
	rf = NotAsRandomForest(n_estimators = t)
	rf.fit(Xtrain, Ytrain)
	our_model_scores[t] = rf.score(Xtest, Ytest)

	bagging = BaggingClassifier(n_estimators = t)
	bagging.fit(Xtrain, Ytrain)
	bagging_scores[t] = bagging.score(Xtest, Ytest)

plt.plot(range(1, N_TREES + 1), our_model_scores, label = 'Our Model')
plt.plot(range(1, N_TREES + 1), bagging_scores, label = 'bagging')
plt.legend()
plt.show()










