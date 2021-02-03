import numpy as np
import matplotlib.pyplot as plt

# For dimensionality reduction and visualization
from sklearn.manifold import TSNE
# For clustering
from sklearn.mixture import GaussianMixture

def purity(Y, R):
    # maximum purity is 1, higher is better
    N, K = R.shape
    p = 0
    for k in range(K):
        best_target = -1 # we don't strictly need to store this
        max_intersection = 0
        for j in range(K):
            intersection = R[Y==j, k].sum()
            if intersection > max_intersection:
                max_intersection = intersection
                best_target = j
        p += max_intersection
    return p / N

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
    X, Y = get_mnist(limit = 1000)

    tsne = TSNE()
    Z = tsne.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    # purity measure from unsupervised machine learning pt 1
    # maximum purity is 1, higher is better
    gmm = GaussianMixture(n_components=10)
    gmm.fit(X)
    R = gmm.predict_proba(X)
    print("Purity Score: ", purity(Y, R))

    # now try the same thing on the reduced data
    gmm.fit(Z)
    R = gmm.predict_proba(Z)
    print("Purity Score: ", purity(Y, R))

if __name__ == '__main__':
    main()