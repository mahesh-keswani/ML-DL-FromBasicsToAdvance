import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cross_entropy(probs, target):
	return -np.mean(target*np.log(probs) + (1 - target)*np.log(1 - probs))

N = 1000
D = 2

radius_inner = 5
radius_outer = 10

# creates vector of radii for the inner circle
R1 = np.random.randn(N // 2) + radius_inner
# creates a vector of thetas for inner circle
theta = 2 * np.pi * np.random.random(N // 2)

# x = r * cos(theta), y = r * sin(theta)
# Creating X_inner shape (500x2)
X_inner = np.concatenate([ [R1*np.cos(theta)], [R1*np.sin(theta)] ]).T

# creates vector of radii for the inner circle
R2 = np.random.randn(N // 2) + radius_outer
# creates a vector of thetas for inner circle
theta = 2 * np.pi * np.random.random(N // 2)

# x = r * cos(theta), y = r * sin(theta)
# Creating X_inner shape (500x2)
X_outer = np.concatenate([ [R2*np.cos(theta)], [R2*np.sin(theta)] ]).T

# Final X
X = np.concatenate( [X_inner, X_outer] )

# creating targets
T = np.array([0]*(N // 2) + [1]*(N // 2))

plt.scatter(X[:, 0], X[:, 1], c=T, s=100)
plt.show()

# Now solving using logistic regression
bias = np.ones((N, 1))

# adding the radius of every sample
r = np.zeros((N, 1))
for i in range(N):
	# r = np.sqrt(x^2 + y^2)
	r[i] = np.sqrt(X[i, :].dot(X[i, :]))

Xb = np.concatenate((X, r, bias), axis = 1)

# using logistics regression
w = np.random.randn(D + 2) / np.sqrt(D + 2)
learning_rate = 0.0001
costs = []
accuracy = []
for i in range(1000):
	yhat = sigmoid(Xb.dot(w))

	error = cross_entropy(yhat, T)
	costs.append(error)

	diff = yhat - T

	predictions = np.round(yhat)
	acc = np.mean(predictions == T)
	accuracy.append(acc)

	if i % 10 == 0:
		print(acc)

	w = w - learning_rate*(Xb.T.dot(diff))

plt.plot(costs, label='cost')
plt.plot(accuracy, label='accuracy')
plt.legend()
plt.show()
