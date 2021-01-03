import numpy as np 
import matplotlib.pyplot as plt 

from scipy.cluster.hierarchy import dendrogram, linkage

def main():

	D = 2
	N = 900
	s = 5
	X = np.zeros((N, D))
	mu1 = np.array([[0, 0]])
	mu2 = np.array([[s, s]])
	mu3 = np.array([[-s, -s]])
	
	X[:300, :] = np.random.randn(300, D) + mu1
	X[300:600, :] = np.random.randn(300, D) + mu2
	X[600:, :] = np.random.randn(300, D) + mu3
	
	plt.scatter(X[:, 0], X[:, 1])
	plt.show()

	Z = linkage(X, 'ward')
	print("Z.shape:", Z.shape)
	print("First five rows...")
	print(Z[:5])
    # Z has the format [idx1, idx2, dist, sample_count]
    # therefore, its size will be (N-1, 4)

	plt.title("Ward")
	dendrogram(Z)
	plt.show()

	Z = linkage(X, 'single')
	plt.title("Single")
	dendrogram(Z)
	plt.show()

	Z = linkage(X, 'complete')
	plt.title("Complete")
	dendrogram(Z)
	plt.show()

if __name__ == '__main__':
    main()