import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse 

NUM_DATASETS = 50
VAR_OF_NOISE = 0.5
MAX_POLY = 12
# NUMBER OF SAMPLES IN EACH DATASET
N = 25 
train_size = int(0.9 * N)

def make_poly(x, degree):
	# x is vector of len n
	N = len(x)
	# +1 for bias, the first column will be all ones
	newX = np.empty((N, degree + 1))
	for d in range(degree + 1):
		newX[:, d] = x**d 
		# normalizing for d > 1
		if d > 1:
			newX[:, d] = (newX[:, d] - newX[:, d].mean()) / newX[:, d].std()

	return newX

def f(X):
	return np.sin(X)

# preparing dataset
X = np.linspace(-np.pi, np.pi, N)
np.random.shuffle(X)

Xpoly = make_poly(X, MAX_POLY)
f_X = f(X)

# for plotting
train_scores = np.zeros((NUM_DATASETS, MAX_POLY))
test_scores = np.zeros((NUM_DATASETS, MAX_POLY))

train_predictions = np.zeros((train_size, NUM_DATASETS, MAX_POLY))

model = LinearRegression()
for k in range(NUM_DATASETS):
	# y = f(x) + noise
	Y = f_X + np.random.randn()*VAR_OF_NOISE

	Xtrain = Xpoly[:train_size]
	Ytrain = Y[:train_size]

	Xtest = Xpoly[train_size:]
	Ytest = Y[train_size:]

	for degree in range(MAX_POLY):
		# +2 because one for bias and one since range will work till MAX_POLY - 1
		model.fit(Xtrain[:, :degree + 2], Ytrain)

		predictions = model.predict(Xpoly[:, :degree + 2])

		train_prediction = predictions[:train_size]
		test_prediction = predictions[train_size:]

		train_score = mse(train_prediction, Ytrain)
		test_score = mse(test_prediction, Ytest)

		train_scores[k, degree] = train_score
		test_scores[k, degree] = test_score

		train_predictions[:, k, degree] = train_prediction

# Finding bias 
avg_train_prediction_over_datasets = np.zeros((train_size, MAX_POLY))
squared_bias = np.zeros(MAX_POLY)
true_f_x = f_X[:train_size]

for degree in range(MAX_POLY):
	for k in range(train_size):
		# taking the average for kth sample in all the datasets for all degrees
		avg_train_prediction_over_datasets[k, degree] = train_predictions[k, : , degree].mean()

	# bias = (true_f_x - avg_prediction of kth sample over all datasets for all degree)**2
	# then mean
	squared_bias[degree] = ((true_f_x - avg_train_prediction_over_datasets[:, degree])**2).mean()


# Now variance
variances = np.zeros((train_size, MAX_POLY))
for degree in range(MAX_POLY):
	for k in range(train_size):
		# delta = true_f_x - f_bar_x ( avg of prediction of kth sample over degree d )
		delta = true_f_x - avg_train_prediction_over_datasets[k, degree] 

	variances[k, degree] = delta.dot(delta) / train_size

# now variance is vector of len MAX_POLY
variance = variances.mean(axis = 0)

# Now plotting
degrees = np.arange(MAX_POLY) + 1
best_degree = np.argmin(test_scores.mean(axis = 0))

plt.plot(degrees, squared_bias, label = 'squared bias')
plt.plot(degrees, variance, label = 'variance')
plt.plot(degrees, squared_bias + variance, label = 'squared bias + variance')
plt.axvline(best_degree, linestyle = '--', label = 'best_degree')
plt.legend()
plt.show()

plt.plot(degrees, train_scores.mean(axis = 0), label = 'train_scores')
plt.plot(degrees, test_scores.mean(axis = 0), label = 'test_scores')
plt.axvline(best_degree, linestyle = '--', label = 'best_degree')
plt.legend()
plt.show()



















































