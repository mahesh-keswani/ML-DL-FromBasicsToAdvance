import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(probs, target):
	return -np.mean(target*np.log(probs) + (1 - target)*np.log(1 - probs)) + l1*np.abs(w).mean()

N = 50
D = 50

# data centered around 0 from [-5, 5]
X = (np.random.random((N, D)) - 0.5)*10
true_w = np.array([1, 0.5, -0.5] + [0]*(D - 3))

y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))

w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 2
costs = []
for i in range(100):
	yhat = sigmoid(X.dot(w))
	# Note: This has to be yhay - y and not reverse else w will not get 
	#       optimized. 
	diff = yhat - y
	error = cross_entropy(yhat, y)
	costs.append(error)
	w = w - learning_rate*(X.T.dot(diff) + l1*np.sign(w))

plt.plot(costs)
plt.show()

plt.plot(true_w, label='true_w')
plt.plot(w, label='w calculated')
plt.legend()
plt.show()






