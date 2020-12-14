from basic_kmeans import plot_kmeans
import numpy as np 
import matplotlib.pyplot as plt 
from util import donut, get_elliptical_distributions, imbalanced_normal_distribution

def main():
	X, Y = donut()

	plt.scatter(X[:, 0], X[:, 1], c=Y, s=100)
	plt.show()

	plot_kmeans(X, 2, show_plot = True)

	X = get_elliptical_distributions()
	plt.scatter(X[:, 0], X[:, 1], s=100)
	plt.axis('off') 
	plt.show()

	plot_kmeans(X, 2, show_plot = True)

	X = imbalanced_normal_distribution()
	plt.scatter(X[:, 0], X[:, 1], s=100)
	plt.axis('off') 
	plt.show()
	
	plot_kmeans(X, 2, show_plot = True)

if __name__ == '__main__':
	main()