import numpy as np 
import pandas as pd 

def get_data():
	df = pd.read_csv('ecommerce_data.csv')
	time_of_day_dummies = pd.get_dummies(df['time_of_day'])

	df.drop(['time_of_day'], axis = 1, inplace = True)
	df = pd.concat([df, time_of_day_dummies], axis = 1)
	
	Y = df['user_action'].values
	X = df.drop(['user_action'], axis = 1).values

	return X, Y

def get_binary_data():
	X, y = get_data()
	
	# getting only X's where y = 0 and y = 1
	X = X[y <= 1]
	y = y[y <= 1]

	return X, y