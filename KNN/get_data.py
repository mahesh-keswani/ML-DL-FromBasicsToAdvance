import numpy as np 

def get_mnist(limit = None):
	X = []
	Y = []

	firstLine = True
	for count, line in enumerate(open('../Data/train.csv')):

		if firstLine:
			firstLine = False
		else:
			line = line.split(",")
			y = float(line[0])
			x = [float(i) for i in line[1:]]

			X.append(x)
			Y.append(y)
		if count == limit: 
			break

	X = np.array(X) / 255.0
	Y = np.array(Y)

	print(X.shape, Y.shape)
	return X, Y

def get_xor():
	X = np.zeros((200, 2))
	# (0.5 - 1, 0.5 - 1)
	X[:50] = np.random.random((50, 2)) / 2 + 0.5 

	# (0 - 0.5, 0 - 0.5)
	X[50:100] = np.random.random((50, 2)) / 2

	# (0 - 0.5, 0.5 - 1)
	X[100:150] = np.random.random((50, 2)) / 2 + np.array([ [0, 0.5] ])

	# (0.5 - 1, 0 - 0.5)
	X[150:200] = np.random.random((50, 2)) / 2 + np.array([ [0.5, 0] ])

	Y = np.array([0]*100 + [1]*100)
	return X, Y

def get_donut():
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

	return X, T











