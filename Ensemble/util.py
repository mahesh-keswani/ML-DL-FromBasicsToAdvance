from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

# code source for decision boundry: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/supervised_class2/util.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def plot_decision_boundary(X, model):
  h = .02  # step size in the mesh
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))


def get_housing_data(train_size = 0.9):
	path = "../Data/hou_all.csv"
	df = pd.read_csv(path, header = None)

	X, Y = df.values[:, :-2], df.values[:, -2]
	N = len(X)

	# adding bias
	X = np.concatenate(( X, np.ones((N, 1)) ), axis = 1)

	Xtrain, Ytrain = X[:int(train_size * N)], Y[:int(train_size * N)]
	Xtest, Ytest = X[int(train_size * N):], Y[int(train_size * N):]

	numeric_cols = list(range(14))
	numeric_cols.pop(3)

	sc = StandardScaler()
	sc.fit(Xtrain[:, numeric_cols])

	Xtrain[:, numeric_cols] = sc.transform(Xtrain[:, numeric_cols])
	Xtest[:, numeric_cols] = sc.transform(Xtest[:, numeric_cols])

	return Xtrain, Xtest, Ytrain, Ytest


def get_xor(train_size = 0.9):

	N = 500
	D = 2

	X = np.random.randn(N, D)
	seperator = 2

	X[:125] += np.array([seperator, seperator])
	X[125:250] += np.array([seperator, -seperator])
	X[250:375] += np.array([-seperator, seperator])
	X[375:] += np.array([-seperator, -seperator])

	Y = np.array([1]*125 + [0] * 250 + [1] * 125)

	train_size = int(train_size * N)
	Xtrain, Ytrain = X[:train_size], Y[:train_size]
	Xtest, Ytest = X[train_size:], Y[train_size:]

	return Xtrain, Ytrain, Xtest, Ytest

























