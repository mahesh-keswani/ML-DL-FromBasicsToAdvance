from util import get_simple_data
from basic_kmeans import plot_kmeans, cost 
import numpy as np 
import matplotlib.pyplot as plt 

def main():
	X = get_simple_data()

	plt.scatter(X[:, 0], X[:, 1], s=100)
	plt.show()

	costs = np.empty(10)
	costs[0] = None
	for k in range(1, 10):
		M, R = plot_kmeans(X, k, show_plot = False)
		costs[k] = cost(X, R, M)

	plt.plot(costs)
	plt.title("K v/s Loss")
	plt.show()

if __name__ == '__main__':
	main()