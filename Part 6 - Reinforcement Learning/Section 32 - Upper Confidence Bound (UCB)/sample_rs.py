# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 21:14:45 2019

@author: Shubham
"""

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing random selection
import random
ads_selected = []
N = 10000
d = 10
total_reward = 0
for i in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[N,ad]
    total_reward = total_reward + reward

#visualisation
plt.hist(ads_selected)
plt.title('random selection')
plt.xlabel('qd')
plt.ylabel('no. of times ad displayed')
plt.show()