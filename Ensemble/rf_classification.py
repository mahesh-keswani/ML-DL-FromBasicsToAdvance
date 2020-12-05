'''
	Doing comparison of random forest, logistic regression, decision tree classifier,
	bagged tree classifier
'''
from util import get_xor
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 

Xtrain, Ytrain, Xtest, Ytest = get_xor()
plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain, s=100)
plt.show()

def fit_model_and_show_score(models, Xtrain, Ytrain, Xtest, Ytest, names):
	for model, name in zip(models, names):
	
		print("Score for {} is {}".format(name, cross_val_score(model, Xtrain, Ytrain, cv = 5).mean()))

rf = RandomForestClassifier(n_estimators = 100)
linear_model = LogisticRegression()
decision_tree = DecisionTreeClassifier()
bagged_model = BaggingClassifier(n_estimators = 100)

models = [rf, linear_model, decision_tree, bagged_model]
names = ['RandomForestClassifier', 'LogisticRegression', 'DecisionTreeClassifier', 'BaggingClassifier']

fit_model_and_show_score(models, Xtrain, Ytrain, Xtest, Ytest, names)