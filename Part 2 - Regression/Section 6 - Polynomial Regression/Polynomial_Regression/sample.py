# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 21:43:33 2019

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
y = dataset.iloc[:,2]

#ploating data
plt.scatter(X,y,color='red')
plt.title('level vs salary')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly,y)
y_pred = regressor.predict(X_poly)

#ploating data

#smooth curve
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('level vs salary')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


