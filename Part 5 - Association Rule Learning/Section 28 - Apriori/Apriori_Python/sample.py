# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:46:02 2019

@author: Shubham
"""
#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transaction = []
for i in range(0,7501):
    transaction.append([ str(dataset.values[i,j]) for j in range(0,20)])

#training
from apyori import apriori
rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


#visualisation
result = list(rules)