import numpy as np 
import matplotlib.pyplot as plt 


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(probs, target):
	return -np.mean(target*np.log(probs) + (1 - target)*np.log(1 - probs))

N = 4
D = 2

X = np.array([
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1]
])

T = np.array([0, 1, 1, 0])

plt.scatter(X[:, 0], X[:, 1], c = T, s=100)
plt.show()

# adding bias
bias = np.ones((N, 1))

# also adding product of first dimension and second dimension of x 
xy = np.array([ X[:, 0]*X[:, 1] ]).T

X = np.concatenate((X, bias, xy), axis = 1)

# Now solving using logistic regression
w = np.random.randn(D + 2) / np.sqrt(D)
learning_rate = 0.0001
costs = []
for i in range(100):
	yhat = sigmoid(X.dot(w))
	diff = yhat - T

	error = cross_entropy(yhat, T)
	costs.append(error)

	w = w - learning_rate*(X.T.dot(diff))

plt.plot(costs)
plt.show()




