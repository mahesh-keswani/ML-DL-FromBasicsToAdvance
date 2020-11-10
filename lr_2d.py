import numpy as np 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection="3d")

plt.show()
# load the data
X = []
Y = []
for line in open('data_2d.csv'):
	x1, x2, y = line.split(",")
	x1 = float(x1)
	x2 = float(x2)
	y = float(y)

	X.append([x1, x2, 1])
	Y.append(y)

x = np.array(X)
y = np.array(Y)

# Plotting data
fig = plt.figure()
ax = plt.axes(projection="3d")

z_line = y
x_line = x[:, 0]
y_line = x[:, 1]
ax.scatter3D(x_line, y_line, z_line, color = "green")
plt.show()

# Now calculating weights of x's and return r squared
def return_r2(x, y):
	#            2x100 100x2 => (2,2)
	xTx = np.dot(x.T, x)
	print("xTx shape: ", xTx.shape)
	#            2x100 100x1 => (2,) 
	xTy = np.dot(x.T, y)
	print("xTy shape: ", xTy.shape)
	#                   (2, 2)matrix * w = (2,) vector
	w = np.linalg.solve(xTx, xTy)

	# w shape = (2,)
	# calculating yhat
	#             100x2 2x1 => (100, 1) = yhat
	yhat = np.dot(x, w)

	# calculating r2

	residuals = y - yhat
	total = y - y.mean()

	r2 = 1 - (residuals.dot(residuals) / total.dot(total))
	return r2


print("r2 using x1 only: {}\n\n".format(return_r2(x[:, [0, 2]], y )))
print("r2 using x2 only: {}\n\n".format(return_r2(x[:, [0, 1]], y )))
print("r2 using both: {}\n\n".format(return_r2(x, y )))
