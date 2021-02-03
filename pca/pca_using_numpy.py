import numpy as np
import matplotlib.pyplot as plt

from get_data import get_mnist

# get the data
X, Y = get_mnist(limit = 1000)

# decompose covariance
covX = np.cov(X.T.dot(X))
# getting eigenvalues and eigenvectors
lambdas, Q = np.linalg.eigh(covX)


# lambdas are sorted by default from smallest --> largest
# some may be slightly negative due to precision
idx = np.argsort(-lambdas)
lambdas = lambdas[idx] # sort in proper descending order
lambdas = np.maximum(lambdas, 0) # get rid of negatives
Q = Q[:,idx]


# plot the first 2 columns of Z, getting the data into new basis
Z = X.dot(Q)
plt.scatter(Z[:,0], Z[:,1], s=100, c=Y, alpha=0.3)
plt.show()


# plot variances
plt.plot(lambdas)
plt.title("Variance of each component")
plt.show()

# cumulative variance
plt.plot(np.cumsum(lambdas))
plt.title("Cumulative variance")
plt.show()