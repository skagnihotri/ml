# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:08:38 2019

@author: Shubham
"""

#lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing ucb
import math
N = 10000
d = 10
ads_selection = []
no_of_selection = [0]*d
sum_of_rewards = [0]*d
total_reward =0
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if no_of_selection[i]>0:
            average_reward = sum_of_rewards[i]/no_of_selection[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/no_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound>max_upper_bound:
            max_upper_bound=upper_bound
            ad=i
    ads_selection.append(ad)
    no_of_selection[ad]=no_of_selection[ad]+1
    reward = dataset.values[n,ad]
    sum_of_rewards[ad]=sum_of_rewards[ad]+reward
    total_reward=total_reward+reward
    
#visualisation
plt.hist(ads_selection)
plt.title('UCB')
plt.xlabel('ad')
plt.ylabel('no . of ads')
plt.show()
        
        
        
        