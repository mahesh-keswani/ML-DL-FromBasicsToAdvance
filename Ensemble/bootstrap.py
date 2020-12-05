import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm

# no of bootstrap samples
B = 200
N = 20
X = np.random.randn(N)

print("Sample mean: {}".format(X.mean()))

individual_estimates = np.empty(B)
for b in range(B):
	# Note: sample with replacement is true by default
	sample = np.random.choice(X, size = N)
	individual_estimates[b] = sample.mean()

bootstraped_mean = individual_estimates.mean()
bootstraped_std = individual_estimates.std()

# calculating confidence intervals for 95% using estimates
# norm ppf returns the value of x for which AUC is 0.025
bootstrapped_lower_limit = bootstraped_mean + norm.ppf(0.025) * bootstraped_std
bootstrapped_upper_limit = bootstraped_mean + norm.ppf(0.975) * bootstraped_std

print("bootstraped_mean", bootstraped_mean)

# std_of_population ~= std_of_sample / sqrt(sample size)
actual_lower_limit = X.mean() + norm.ppf(0.025) * X.std() / np.sqrt(N)
actual_upper_limit = X.mean() + norm.ppf(0.975) * X.std() / np.sqrt(N)


plt.axvline(bootstrapped_lower_limit, linestyle='--', c = 'g', label = 'bootstraped_mean')
plt.axvline(actual_lower_limit, linestyle='--', c = 'r',label = 'actual_lower_limit')
plt.axvline(bootstrapped_upper_limit, linestyle='--', c = 'blue',label = 'bootstrapped_upper_limit')
plt.axvline(actual_upper_limit, linestyle='--', c = 'orange',label = 'actual_upper_limit')
plt.legend()
plt.show()











