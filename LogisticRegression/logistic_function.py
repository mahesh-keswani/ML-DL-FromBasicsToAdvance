import numpy as np 

N = 100
D = 2

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# variance 1 and mean at (2, 2)
X = np.random.randn(N, D) + 2*np.ones((N, D))
bias = np.ones((N, 1))

X = np.concatenate((bias, X), axis = 1)
w = np.random.randn(D + 1)

z = X.dot(w)
print(sigmoid(z))