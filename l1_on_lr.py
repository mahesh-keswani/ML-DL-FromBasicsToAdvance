import numpy as np 
import matplotlib.pyplot as plt 

N = 50
D = 50

# centered around zero from -5 to +5
X = (np.random.random((N, D)) - 0.5)*10

true_w = np.array([1, 0.5, -0.5] + [0]*(D - 3))

true_y = X.dot(true_w) + np.random.randn(N)

# now randomly initializing weights 
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 20

costs = []
for i in range(500):
	yhat = X.dot(w)
	# Note: This has to be yhat - true_y and not reverse else it will 
	# create problems.
	diff = yhat - true_y
	error = diff.dot(diff) / N
	costs.append(error)

	w = w - learning_rate*(X.T.dot(diff) + l1*np.sign(w))

plt.plot(costs)
plt.show()

plt.plot(true_w, label='true_w')
plt.plot(w, label='calculated_w')
plt.legend()
plt.show()

print("Weights calculated", w)