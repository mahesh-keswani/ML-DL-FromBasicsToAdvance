from keras.models import Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Input

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from get_data import getMNIST3D

# get the data
Xtrain, Ytrain, Xtest, Ytest = getMNIST3D()

# get shapes
N, H, W, C = Xtrain.shape
K = len(set(Ytrain))

# make the CNN
i = Input(shape=(H, W, C))
x = Conv2D(filters=32, kernel_size=(3, 3))(i)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=64, kernel_size=(3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dense(units=100)(x)
x = Activation('relu')(x)
x = Dense(units=K)(x)
x = Activation('softmax')(x)

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