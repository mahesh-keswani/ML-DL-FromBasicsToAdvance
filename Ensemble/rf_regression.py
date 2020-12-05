'''
	Doing comparison of random forest, linear regression, decision tree regressor,
	bagged tree regressor
'''
from util import get_housing_data
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt 

Xtrain, Xtest, Ytrain, Ytest = get_housing_data()
Ytrain = np.log(Ytrain)
Ytest = np.log(Ytest)

def fit_model_and_show_plot(models, Xtrain, Ytrain, Xtest, Ytest, names):
	for model, name in zip(models, names):
		model.fit(Xtrain, Ytrain)
		predictions = model.predict(Xtest)

		plt.plot(predictions, label = 'predictions from {}'.format(name))
	
		print("Score for {} is {}".format(name, model.score(Xtest, Ytest)))

	plt.plot(Ytest, label = 'targets')
	plt.legend()
	plt.show()


rf = RandomForestRegressor(n_estimators = 500)
linear_model = LinearRegression()
decision_tree = DecisionTreeRegressor()
bagged_model = BaggingRegressor(n_estimators = 500)

models = [rf, linear_model, decision_tree, bagged_model]
names = ['RandomForestRegressor', 'LinearRegression', 'DecisionTreeRegressor', 'BaggingRegressor']

fit_model_and_show_plot(models, Xtrain, Ytrain, Xtest, Ytest, names)