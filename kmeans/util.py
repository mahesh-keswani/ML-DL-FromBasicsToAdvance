import numpy as np 
import matplotlib.pyplot as plt 

def donut():
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

def get_elliptical_distributions():

	X = np.zeros((1000, 2))
	# parameters: mean vector of len 2 (Dimension of X), Covariance matrix of DxD, no_of_samples
	X[:500, :] = np.random.multivariate_normal([0, 0], [ [1, 0],[0,20] ], 500)
	X[500:, :] = np.random.multivariate_normal([5, 0], [ [1, 0],[0,20] ], 500)

	return X

def imbalanced_normal_distribution():
	X = np.zeros((1000, 2))

	X[:950, :] = np.array([0, 0]) + np.random.randn(950, 2)
	X[950:, :] = np.array([5, 0]) + np.random.randn(50, 2)
	
	return X

def get_simple_data():
	s = 4
	D = 2

	mu1 = np.array([0, 0])
	mu2 = np.array([s, s])
	mu3 = np.array([0, s])
	
	N = 900
	X = np.zeros((N, D))
	X[:300, :] = np.random.randn(300, D) + mu1
	X[300:600, :] = np.random.randn(300, D) + mu2
	X[600:, :] = np.random.randn(300, D) + mu2

	return X