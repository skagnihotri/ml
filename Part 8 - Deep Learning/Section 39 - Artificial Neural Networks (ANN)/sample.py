# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:38:18 2019

@author: Shubham
"""

# ANN model

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

#categorical variables
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_X1 = LabelEncoder()
X[:,1] = labelencoder_X1.fit_transform(X[:,1])
labelencoder_X2 = LabelEncoder()
X[:,2] = labelencoder_X2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#spliting into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#making ANN model

#importing lib and packages
import keras