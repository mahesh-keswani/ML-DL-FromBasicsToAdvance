import numpy as np 

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
