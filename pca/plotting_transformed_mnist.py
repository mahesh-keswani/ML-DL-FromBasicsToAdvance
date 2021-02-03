import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from get_data import get_mnist

def main():
	X, Y = get_mnist(limit = 1000)

	pca = PCA()
	reduced = pca.fit_transform(X)

	# generating random number from 1 to 1000
	random_number = np.random.randint(low = 1, high = 1000)
	random_image = reduced[random_number]

	# plotting original image
	plt.imshow(X[random_number].reshape( (28, 28) ), cmap='gray')
	plt.show()

	# taking top 196 dimensions and reshaping them as 14x14
	plt.imshow(random_image[:196].reshape( (14, 14) ), cmap='gray')
	plt.show()


if __name__ == '__main__':
    main()