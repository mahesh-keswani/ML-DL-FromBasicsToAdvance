import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('nature.jpg')

# what does it look like?
plt.imshow(img)
plt.show()

# taking the mean along the color channel and therefore converting into 2d
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()

# create a Gaussian filter
W = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2
        W[i, j] = np.exp(-dist / 50.)

W /= W.sum() # normalize the kernel

# let's see what the filter looks like
plt.imshow(W, cmap='gray')
plt.show()

# now the convolution
out = convolve2d(bw, W)
plt.imshow(out, cmap='gray')
plt.show()

# after convolution, the output signal is N1 + N2 - 1
print(out.shape)

# we can also just make the output the same size as the input
out = convolve2d(bw, W, mode='same')
plt.imshow(out, cmap='gray')
plt.show()
print(out.shape)








