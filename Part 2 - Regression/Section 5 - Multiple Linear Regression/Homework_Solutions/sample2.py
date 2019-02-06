# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 19:30:45 2019

@author: Shubham
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing databases
dataset = pd.read_csv('50_Startups.csv')
#features
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4]

#categorical values like country has 3 category
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#spliting into train set and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

X=X[:,1:]
#Multivalue Linear Regression
#backward ellimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
# 1
x_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
# 2
x_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
# 3
x_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
# 4
x_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary() #values are near 5% aprrox

#training and testing sets and final predicsting
X_train,X_test,y_train,y_test = train_test_split(x_opt,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)