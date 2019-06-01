# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:38:03 2019

@author: Shubham
"""

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[ : , : -1].values
y = dataset.iloc[:, -1].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

#som
