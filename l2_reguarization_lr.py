import numpy as np 
import matplotlib.pyplot as plt 

# No. of samples
N = 100
X = np.linspace(0, 10, N)
# y = 0.5x + noise
Y = 0.5*X + np.random.randn()

# manually creating the outliers
Y[-1] += 30
Y[-2] += 30

# adding bias to X
X = np.vstack([X, np.ones(N)]).T

plt.scatter(X[:, 0], Y)
plt.show()

# now using maximum likelikhood
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
yhat_ml = np.dot(X, w_ml)

# now using maximum posteriori
yhat_maps = []
lambdas = []
for l2 in np.linspace(100, 2000, 10):
	w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
	yhat_maps.append( np.dot(X, w_map) )
	lambdas.append(l2)

# Now plotting everythig
plt.scatter(X[:, 0], Y)
plt.plot(X[:, 0], yhat_ml, label="Without l2")

for i, yhat_map in enumerate(yhat_maps):
	plt.plot(X[:, 0], yhat_map, label="With lambda: {}".format(lambdas[i]))

plt.legend()
plt.show()