import numpy as np 
import matplotlib.pyplot as plt 

# loading the data
X = []
Y = []
for line in open('data_1d.csv'):
	x, y = map(float, line.split(','))
	random = np.random.uniform(low = 0, high = 100)

	# Features: [x, noise]
	X.append([ x, random ])
	Y.append(y)

x = np.array(X)
y = np.array(Y)

# calculating weights
w = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))

# now yhats
yhats = np.dot(x, w)

# now r-squared
residual = y - yhats
total = y - y.mean()

r2 = 1 - ( np.dot(residual, residual) / np.dot(total, total) )
print("R2: ", r2)
