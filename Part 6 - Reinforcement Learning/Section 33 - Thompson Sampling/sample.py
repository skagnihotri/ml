# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:28:10 2019

@author: Shubham
"""

#libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing thompson sampling
import random
N=10000
d=10
total_reward = 0
ads_selected =[]
no_of_reward1= [0]*d
no_of_reward0= [0]*d
for n in range(0,N):
    ad=0
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(no_of_reward1[i]+1,no_of_reward0[i]+1)
        if random_beta>max_random:
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    total_reward=total_reward+reward
    if reward==1:
        no_of_reward1[ad]=no_of_reward1[ad]+1
    else:
        no_of_reward0[ad]=no_of_reward0[ad]+1

#visualisation
plt.hist(ads_selected)
plt.title('thompson sampling')
plt.xlabel('ad')
plt.ylabel('no. od ads')
plt.show()
        
        
        
        