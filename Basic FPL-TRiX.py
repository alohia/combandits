#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:11:38 2017

@author: angus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
truncexp = stats.truncexpon
# Kveton et al 2015 CombUCB1 algorithm
# regret should be bounded above by O(KL(1/∆) log n) and below by Ω(KL(1/∆) log n)

# Scenario: User picks K features from a vector of length L
# At each play the return from the selected elements of the vector is drawn from bernoulli
# distribution with increasing means (so later entries in the vector are more likely to return 1
# and earlier entries more likely to return 0)
# Algorithm attempts to maximise the return (so should learn to pick the entries at the end of the vector)

### Set parameters
L = 40
K = 8
iters = 10000

# model params
eta = 1 # this acts against the perturbations, so the higher this is the less significant the perturbations in the combinatorial optimisation
gamma = 0.1 # the higher this is the more the alorithm explores
Bt = 1

### Set arrays to store at each iteration
# perturbations vector (Z)
# the K elements selected to play in the next round (based on the upper confidence bounds)
# losses/weights drawn for the selected elements (w) and 0 for non-observed
# l_hat = w/(E(w_t|F_(t-1))+gamma)
# cumulative l_hats (L_hat)

Z = np.zeros((L, iters))
A = np.zeros((L, iters))
w = np.zeros((L, iters))
l_hat = np.zeros((L, iters))
L_hat = np.zeros((L, iters))

### set means of each element
norm_means = pd.Series(list(range(L)))

### Run algorithm from t=1,,, (start at 1 as L_hat is initialised as 0s, so 0th column is this init)
for t in range(iters)[1:]:
    
    # draw perturbation vector
    Z[:,t] = truncexp.rvs(L)
    
    # select best action to minimise expected cost
    A[np.argsort(eta*L_hat[:,(t-1)] - Z[:,t])[:K],t] = 1
    
    # draw weights for chosen actions
    w[:,t] = A[:,t]*np.random.normal(norm_means)
    
    # calculate l_hats
    l_hat[:,t] = w[:,t]/(np.sum(A, axis=1)/t + gamma)
    
    # update L_hats
    L_hat[:,t] = L_hat[:,(t-1)] + l_hat[:,t]



# plot development of selected elements
plt.pcolor(A[:,1:])


# Calculate regret bounds and test parameters



