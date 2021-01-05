import numpy as np 
import matplotlib.pyplot as plt 

N = 1500
s = 4
D = 2
K = 3

X = np.zeros((N, D))
X[:500, :] = np.random.randn(500, D) + np.array([0, 0])
X[500:1000, :] = np.random.randn(500, D) + np.array([s, s])
X[1000:, :] = np.random.randn(500, D) + np.array([-s, -s])

Y = np.array([0]*500 + [1]*500 + [2]*500)

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()

neurons_in_hidden = 3

W1 = np.random.randn(D, neurons_in_hidden)
b1 = np.random.randn(neurons_in_hidden)
W2 = np.random.randn(neurons_in_hidden, K)
b2 = np.random.randn(K)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def forward(X, W1, b1, W2, b2):
	Z = sigmoid(X.dot(W1) + b1)
	O = Z.dot(W2) + b2

	expO = np.exp(O)
	exp_softmax = expO / expO.sum(axis = 1, keepdims = True)

	return exp_softmax

def classification_rate(Y, P):
	return np.mean(Y == P)

P_Y_given_x = forward(X, W1, b1, W2, b2)

predictions = np.argmax(P_Y_given_x, axis = 1)

print(classification_rate(Y, predictions))








