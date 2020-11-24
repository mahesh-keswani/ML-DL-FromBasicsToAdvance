import numpy as np 
import matplotlib.pyplot as plt 
from knn_for_mnist import KNN 

def get_data():
	width = 8
	height = 8
	N = width * height

	X = np.zeros((N, 2))
	Y = np.zeros(N)

	'''
	for shape = (3, 3)
		X  = [ [0, 0]
			   [0, 1]
			   [0, 2]
			   [1, 0]...
			]
		Y = [
				0
				1
				0
				1
				0...
			 ]
	'''
	n = 0
	start_t = 0
	for i in range(height):
		t = start_t
		for j in range(width):
			X[n] = [i, j]
			Y[n] = t
			t = (t + 1) % 2
			n += 1
			
		start_t = (start_t + 1) % 2
	return X, Y

if __name__ == '__main__':
	X, Y = get_data()
	plt.scatter(X[:, 0], X[:, 1], c= Y, s=100)
	plt.show()
	
	knn = KNN(3)
	knn.fit(X, Y)

	print("Train score: {}".format(knn.score(X, Y)))




