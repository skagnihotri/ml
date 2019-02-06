# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:39:36 2019

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

#regressor for desision tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#predict
y_pred = regressor.predict(X)

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()