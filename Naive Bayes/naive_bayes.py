import numpy as np 
from datetime import datetime 
from scipy.stats import multivariate_normal as mvn 
from get_data import get_mnist

class NaiveBayes(object):
	# smoothing for avoiding singular covariance problem
	def fit(self, X, Y, smoothing=10e-3):
		# will be used to store mean and variance for each class
		self.gaussians = {}
		# will be used to store priors for each class i.e P(C)
		self.priors = {}

		all_classes = set(Y)
		for c in all_classes:
			current_x = X[Y == c]
			self.gaussians[c] = {
				'mean':current_x.mean(axis = 0),
				'var':current_x.var(axis = 0) + smoothing
			}
			self.priors[c] = float(len(Y[Y == c])) / len(Y)

	def score(self, X, Y):
		predictions = self.predict(X)
		return np.mean(predictions == Y)


	def predict(self, X):
		N, D = X.shape
		no_of_classes = len(self.gaussians)
		P = np.zeros((N, no_of_classes))

		for c, gaussian in self.gaussians.items():
			mean, var = gaussian['mean'], gaussian['var']
			# passing the variance vector instead of covariance
			# matrix, assuming all features are independent 
			P[:, c] = mvn.logpdf(X, mean = mean, cov = var) + np.log(self.priors[c])
			
		return np.argmax(P, axis = 1) + 1			


if __name__ == '__main__':
	X, Y = get_mnist(limit = 10000)

	Ntrain = 5000
	Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

	nb = NaiveBayes()
	nb.fit(Xtrain, Ytrain)

	t0 = datetime.now()
	print("Training score: {}".format(nb.score(Xtrain, Ytrain)))
	print("Test score: {}".format(nb.score(Xtest, Ytest)))
	print("Took: {}".format(datetime.now() - t0))











