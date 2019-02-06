# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 19:12:06 2019

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

#Multivalue Linear Regression
#all in
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting values
y_pred = regressor.predict(X_test)