import numpy as np 
import matplotlib.pyplot as plt 

# load the data
X = []
Y = []
for line in open('data_poly.csv'):
	x, y = map(float, line.split(","))
	X.append([x, x*x, 1])
	Y.append(y)

# plot the data
x = np.array(X)
y = np.array(Y)

plt.scatter(x[:, 0], y)
plt.show()

# calculating weights
w = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))

# now yhats
yhats = np.dot(x, w)

# now r-squared
residual = y - yhats
total = y - y.mean()

r2 = 1 - ( np.dot(residual, residual) / np.dot(total, total) )
print("R2: ", r2)