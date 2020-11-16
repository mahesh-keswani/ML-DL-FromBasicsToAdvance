import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

N = 100
D = 2

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

'''
	probs -> vector of probs
	targets -> vector of targets
'''

def get_total_error(probs, target):
	return -np.mean(target*np.log(probs) + (1 - target)*np.log(1 - probs))

X = np.random.randn(N, D)

# setting half data as variance 1 and mean (-2, -2)
X[:50, :] -= 2*np.ones((50, D))

# setting remaining half as variance 1 and mean (2, 2)
X[50:, :] += 2*np.ones((50, D))

bias = np.ones((N, 1))

# X -> (x1, x2, bias)
Xb = np.concatenate((X, bias), axis = 1)

y = np.array([0]*50 + [1]*50)

# Plotting data
fig = plt.figure()
ax = plt.axes(projection="3d")

z_line = y
x_line = Xb[:, 0]
y_line = Xb[:, 1]

ax.scatter3D(x_line, y_line, z_line, color = "green")
plt.show()

w = np.random.randn(D) / np.sqrt(D)
costs = []
for i in range(100):
	yhat = sigmoid(Xb.dot(w))
	diff = y






