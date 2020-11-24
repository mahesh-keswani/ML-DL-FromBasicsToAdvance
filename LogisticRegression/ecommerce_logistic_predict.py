import numpy as np 
from process_ecommerce_data import get_binary_data

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

X, y = get_binary_data()
N, D = X.shape

bias = np.ones((N, 1))
Xb = np.concatenate((X, bias), axis = 1)

w = np.random.randn(D + 1)

P_Y_given_x = sigmoid(Xb.dot(w))

predictions = np.round(P_Y_given_x)

accuracy = np.mean(y == predictions)
print("Accuracy",accuracy)