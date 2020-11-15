import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

N = 100
D = 2

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

'''
	y -> individual probability
	target -> individual target
'''
def cross_entropy(y, target):
	if target == 1:
		return -np.log(y)
	else:
		return -np.log(1 - y)

'''
	probs -> vector of probs
	targets -> vector of targets
'''

def get_total_error(probs, target):
	E = 0
	for i in range(N):
		E += cross_entropy(probs[i], target[i])
	return E

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

# performing calculations
W = np.random.randn(D + 1)
P_Y_given_x = sigmoid(Xb.dot(W))

error = get_total_error(P_Y_given_x, y)
print("Error", error)

# Actual weights for this problem [4, 4, 0] here 0 is weight for bias
W_true = [4, 4, 0]

pred_with_true_w = sigmoid(Xb.dot(W_true))

error_with_true_w = get_total_error(pred_with_true_w, y)
print("Error", error_with_true_w)

