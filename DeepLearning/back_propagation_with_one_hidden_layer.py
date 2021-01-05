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

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def cross_entropy_loss(T, Y):
	# print("Loss cross cross_entropy_loss", T.shape, Y.shape)
	total_cost = T * np.log(Y)
	return total_cost.sum()

def forward(X, W1, b1, W2, b2):
	Z = sigmoid(X.dot(W1) + b1)
	O = Z.dot(W2) + b2

	expO = np.exp(O)
	exp_softmax = expO / expO.sum(axis = 1, keepdims = True)
	# returning softmax output as well as the output of hidden layer for weights updation
	return exp_softmax, Z

def classification_rate(Y, P):
	return np.mean(Y == P)

def derivative_of_w2(Z, Y, T):
	# print("derivative_of_w2 ", Z.shape, Y.shape, T.shape)
	dL_dw2 = Z.T.dot(T - Y)
	return dL_dw2

def derivative_of_w1(X, Z, T, Y, W2):
	# X= NxD, hidden=NxM, T=NxK, outputNxK, W2=MxK
	# print("derivative_of_w1 ",X.shape, Z.shape, T.shape, Y.shape, W2.shape)
	dL_dw1 = (T - Y).dot(W2.T) * Z * (1 - Z)
	dL_dw1 = X.T.dot(dL_dw1)

	return dL_dw1

def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)

def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

N = len(Y)
# turn Y into an indicator matrix (one hot encoder) for training
T = np.zeros((N, K))
for i in range(N):
    T[i, Y[i]] = 1

neurons_in_hidden = 4

W1 = np.random.randn(D, neurons_in_hidden)
b1 = np.random.randn(neurons_in_hidden)
W2 = np.random.randn(neurons_in_hidden, K)
b2 = np.random.randn(K)

learning_rate = 10e-5
costs = []

for i in range(1000):
	output, hidden = forward(X, W1, b1, W2, b2)
	if i % 100 == 0:
		c = cross_entropy_loss(T, output)
		costs.append(c)
		P = np.argmax(output, axis=1)
		r = classification_rate(Y, P)
		print("Epoch", i, "cost:", c, "classification_rate:", r)
	
	gW2 = derivative_of_w2(hidden, T, output)
	gb2 = derivative_b2(T, output)
	gW1 = derivative_of_w1(X, hidden, T, output, W2)
	gb1 = derivative_b1(T, output, W2, hidden)

	W2 += learning_rate * gW2
	b2 += learning_rate * gb2
	W1 += learning_rate * gW1
	b1 += learning_rate * gb1

plt.plot(costs)
plt.show()



