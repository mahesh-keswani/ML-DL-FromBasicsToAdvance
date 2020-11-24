import numpy as np 
import matplotlib.pyplot as plt 

# loading the data
X = []
Y = []
for line in open('data_1d.csv'):
	x, y = line.split(',')
	X.append(float(x))
	Y.append(float(y))

x = np.array(X)
y = np.array(Y)

# plot the data
# plt.scatter(x, y)
# plt.show()

# calculate denominator
denom = (np.mean(x * x) - x.mean()**2)

# calculating slope
a = ( np.mean(x * y) - (y.mean() * x.mean()) ) / denom

# intercept
b = ( np.mean(x * x) * y.mean() - (x.mean() * np.mean(x * y)) ) / denom

yhat = a * x + b

plt.scatter(x, y)
plt.plot(x, yhat)
plt.show()

residuals = y - yhat
total = y - y.mean()

r2 = 1 - (residuals.dot(residuals) / total.dot(total))
print("R2: {}".format(r2))






