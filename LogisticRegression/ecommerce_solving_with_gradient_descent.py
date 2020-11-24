import numpy as np 
import matplotlib.pyplot as plt 
from process_ecommerce_data import get_binary_data
from sklearn.utils import shuffle

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def forward(X, W, b):
	return sigmoid(X.dot(W) + b)

def cross_entropy(ys, targets):
	return -np.mean( (targets*np.log(ys) + (1 - targets)*np.log(1 - ys)))

def accuracy(preds, targets):
	return np.mean(preds == targets)

X, y = shuffle(get_binary_data())

Xtrain = X[:-100]
Ytrain = y[:-100]
Xtest = X[-100:]
Ytest = y[-100:]

bias = 0
D = X.shape[1]

w = np.random.randn(D) / np.sqrt(D)

learning_rate = 0.001
train_costs = []
test_costs = []
for _ in range(10000):
	PYtrain = forward(Xtrain, w, bias)
	PYtest = forward(Xtest, w, bias)

	train_error = cross_entropy(PYtrain, Ytrain)
	test_error = cross_entropy(PYtest, Ytest)

	train_costs.append(train_error)
	test_costs.append(test_error)

	w = w - learning_rate*(Xtrain.T.dot(PYtrain - Ytrain))
	bias = bias - learning_rate*(np.sum(PYtrain - Ytrain))

plt.plot(train_costs, label='train_costs')
plt.plot(test_costs, label='test_costs')
plt.legend()
plt.show()

print("train accuracy", accuracy(np.round(PYtrain), Ytrain))
print("test accuracy", accuracy(np.round(PYtest), Ytest))
