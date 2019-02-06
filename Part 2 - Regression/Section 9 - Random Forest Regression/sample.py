# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:55:03 2019

@author: Shubham
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing databases
dataset = pd.read_csv('Position_Salaries.csv')
#features
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,2].values

#random forrest regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

#predict
y_pred = regressor.predict([[6.5]])
# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()