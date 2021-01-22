import numpy as np
import matplotlib.pyplot as plt

from get_data import get_mnist
from keras.models import Model
from keras.layers import Dense, Activation, Input


# get the data
Xtrain, Ytrain, Xtest, Ytest = get_mnist()

# get shapes
N, D = Xtrain.shape
K = len(set(Ytrain))


# ANN with layers [784] -> [500] -> [300] -> [10]
i = Input(shape=(D,))
x = Dense(500, activation='relu')(i)
x = Dense(300, activation='relu')(x)
x = Dense(K, activation='softmax')(x)

# instantiate the model object
model = Model(inputs=i, outputs=x)


# list of losses: https://keras.io/losses/
# list of optimizers: https://keras.io/optimizers/
# list of metrics: https://keras.io/metrics/
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# gives us back a <keras.callbacks.History object>
r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=15, batch_size=32)
print("Returned:", r)

# print the available keys
# should see: dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
print(r.history.keys())

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()


# make predictions and evaluate
probs = model.predict(Xtest) # N x K matrix of probabilities
Ptest = np.argmax(probs, axis=1)
print("Validation acc:", np.mean(Ptest == Ytest))