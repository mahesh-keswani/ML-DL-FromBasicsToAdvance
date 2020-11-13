import numpy as np 
import matplotlib.pyplot as plt 

# suppose we trying to minimize cost function J = w^2, but gradient descent 
# can be used to maximize as well
# we know the minimum value of this cost function will be zero,
# so let's test this algo

# we will start with random weights and then update in the opposite 
# direction of it's gradient

# Formula: w = w - learning_rate * d (J) / dw
# Therefore for this example, w = w - learning_rate * 2 * w

w = 20
learning_rate = 0.1

for _ in range(100):
	w = w - (learning_rate * 2 * w)
	print(w)
