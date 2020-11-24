from get_data import get_donut 
from knn_for_mnist import KNN 
import numpy as np 
import matplotlib.pyplot as plt 

if __name__ == '__main__':
	X, Y = get_donut()
	plt.scatter(X[:, 0], X[:, 1], c=Y, s=100)
	plt.show()

	knn = KNN(3)
	knn.fit(X, Y)
	print("Train score: {}".format(knn.score(X, Y)))