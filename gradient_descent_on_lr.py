import numpy as np 
import matplotlib.pyplot as plt 

# no. of samples
N = 10
# dimensions
D = 3 

X = np.zeros((N, D))
# adding the bias
X[:, 0] = 1

X[:5, 1] = [1]*5

X[-5:, 2] = [1] * 5

# after this X will be
# 1 1 0
# 1 1 0
# 1 1 0
# 1 1 0
# 1 1 0
# 1 0 1
# 1 0 1
# 1 0 1
# 1 0 1
# 1 0 1

# This example also shows that dummy variable trap can also be avoided 
# using gradient descent
# we cannot use the formula: w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
# because X is a singular matrix, and determinant cannot be calculated

Y = np.array([1]*5 + [0]*5)

# weights with variance = 1 / D
w = np.random.randn(D) / np.sqrt(D)

learning_rate = 0.01
costs = []
for epoch in range(100):
	Yhat = np.dot(X, w)
	diff = Yhat - Y 
	mse = np.dot(diff, diff) / N 
	costs.append(mse)

	w = w - (learning_rate * X.T.dot(diff))

plt.plot(costs, label = 'Cost')
plt.legend()
plt.show()

plt.plot(Y, label = 'Targets')
plt.plot(Yhat, label = 'Prediction')
plt.legend()
plt.show()