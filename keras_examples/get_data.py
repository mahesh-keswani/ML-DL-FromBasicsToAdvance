import numpy as np 
import pandas as pd
from sklearn.utils import shuffle

def get_mnist(limit = None):
	train = pd.read_csv('../Data/train.csv').values.astype(np.float32)
    train = shuffle(train)

    # from start to last limit
    Xtrain = train[:-limit,1:] / 255.0
    Ytrain = train[:-limit,0].astype(np.int32)

    # from last limit to end
    Xtest  = train[-limit:,1:] / 255.0
    Ytest  = train[-limit:,0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest

def getMNIST3D():
    Xtrain, Ytrain, Xtest, Ytest = get_mnist()
    Xtrain = Xtrain.reshape(-1, 28, 28, 1)
    Xtest = Xtest.reshape(-1, 28, 28, 1)
    return Xtrain, Ytrain, Xtest, Ytest
