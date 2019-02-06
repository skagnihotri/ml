# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 20:24:01 2019

@author: Shubham
"""

#simple linear regression

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing databases
dataset = pd.read_csv('Salary_Data.csv')
#features
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1]

#spliting into train set and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting data
y_pred = regressor.predict(X_test)

#ploting training set
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('work exp')
plt.ylabel('salary')
plt.show()

#ploting test set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('work exp')
plt.ylabel('salary')
plt.show()



