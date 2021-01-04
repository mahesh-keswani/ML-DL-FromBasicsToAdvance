import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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


def main():
	X, Y = get_mnist(limit = 10000)
	K = 10

	model = GaussianMixture(n_components=K)

	model.fit(X)
	M = model.means_
	R = model.predict_proba(X)

	predictions = np.argmax(R, axis = 1)

	print(np.mean(predictions == Y))

if __name__ == '__main__':
    main()