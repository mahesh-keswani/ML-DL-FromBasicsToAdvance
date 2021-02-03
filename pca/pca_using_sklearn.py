import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from get_data import get_mnist

def main():
    X, Y = get_mnist(limit = 1000)

    pca = PCA()
    reduced = pca.fit_transform(X)
    
    # Plotting data for 2d
    plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    # explained variance ratio is amount of variance explained by new basis
    # w.r.t original data 
    plt.plot(pca.explained_variance_ratio_)
    plt.show()

    # cumulative variance
    # choose k = number of dimensions that gives us 95-99% variance
    # from this we can decide what should be the optimal k for use case
    # From this plot we can see that using only top 100 to 150 dimensions are necessary 
    cumulative = []
    last = 0
    for v in pca.explained_variance_ratio_:
        cumulative.append(last + v)
        last = cumulative[-1]
    plt.plot(cumulative)
    plt.show()

if __name__ == '__main__':
    main()