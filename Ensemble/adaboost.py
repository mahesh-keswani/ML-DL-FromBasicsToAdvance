'''
	implementing adabosst from scratch (using decision tree from sklearn ;)
	also comparing with bagging classifier
'''
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from util import get_xor

Xtrain, Ytrain, Xtest, Ytest = get_xor()

Ytrain[Ytrain == 0] = -1
Ytest[Ytest == 0] = -1

class Adaboost:

	def __init__(self, M):
		self.M = M

	def fit(self, X, Y):
		self.models = []
		self.alphas = []

		N, D = X.shape
		# starting with equal weights for all samples
		W = np.ones(N) / N

		for m in range(self.M):
			stump = DecisionTreeClassifier(max_depth = 1)
			stump.fit(X, Y, sample_weight = W)

			P = stump.predict(X)
			# weighted error
			error = W.dot(P != Y)
			alpha = 0.5 * (np.log(1 - error) / np.log(error))

			W = W*np.exp(-alpha * P * Y)
			# normalizing weights
			W = W / W.sum() 

			self.models.append(stump)
			self.alphas.append(alpha)

	def predict(self, X):
		# we want accuracy and exponential loss for plotting
		N, D = X.shape
		FX = np.zeros(N)
		for alpha, model in zip(self.alphas, self.models):
			FX += alpha * model.predict(X)
		# predictions, FX (probabilities) for calculating loss
		return np.sign(FX), FX

	def score(self, X, Y):
		predictions, FX = self.predict(X)
		loss = np.exp(-Y * FX).mean()
		return np.mean(predictions == Y), loss

N_TREES = 100
our_model_scores = np.zeros(N_TREES)
bagging_scores = np.zeros(N_TREES)
our_model_loss = np.zeros(N_TREES)
for t in range(1, N_TREES):
	rf = Adaboost(M = t)
	rf.fit(Xtrain, Ytrain)
	our_model_scores[t], our_model_loss[t] = rf.score(Xtest, Ytest)

	bagging = BaggingClassifier(n_estimators = t)
	bagging.fit(Xtrain, Ytrain)
	bagging_scores[t] = bagging.score(Xtest, Ytest)

plt.plot(range(1, N_TREES + 1), our_model_scores, label = 'Our Model predictions')
plt.plot(range(1, N_TREES + 1), bagging_scores, label = 'bagging')
plt.plot(range(1, N_TREES + 1), our_model_loss, label = 'loss of adaboost')

plt.legend()
plt.show()




























