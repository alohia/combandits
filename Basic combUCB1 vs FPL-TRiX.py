#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:11:38 2017

@author: angus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
L = 20
K = 5
iters = 10000

### Set arrays to store at each iteration
# an upper confidence bound for each element (U)
# the 8 elements selected to play in the next round (based on the upper confidence bounds)
# weights drawn for the selected elements (w)
# observed weights (w_observed)

U = np.zeros((L, iters))
A = np.zeros((L, iters))
w = np.zeros((L, iters))
w_observed = np.zeros((L, iters))

### set means of each element
bern_means = pd.Series(list(range(L))).map(lambda x: x/L)

### INIT
# set weights matrix to use for both combUCB1 and FPL-TRiX
for t in range(iters):
    # Draw new weights and store in w
    w[:,t] = np.random.binomial(1, bern_means)

t = 0
# Initial observation of weight for each element
w_observed[:,t] = w[:,t]

# counter for number of observations of each element
c = np.array([1]*L)

### run algorithm over iterations 1 to number of iterations
for t in range(1, iters):
    # Calculate upper confidence bounds
    U[:,t] = np.sum(w_observed, axis=1)/c + np.sqrt(1.5*np.log(t)/c)
    
    # Select 8 highest upper confidence bounds to play next round
    A[np.argsort(U[:,t])[-K:],t] = 1
        
    # Store observed weights (0 otherwise) and add 1 to the counts
    w_observed[:,t] = w[:,t]*A[:,t]
    c[A[:,t]==1] += 1

plt.pcolor(w[:,1:100])
plt.xlabel("iteration")
plt.ylabel("vector element")
plt.title("Weight of each edge at each iteration (100 iterations)")
yellow_patch = mpatches.Patch(color=(1, 1, 0), label='1')
blue_patch = mpatches.Patch(color=(0, 0, 1), label='0')
plt.legend(handles=[yellow_patch, blue_patch], bbox_to_anchor=(1.05, 1), loc=2)

# plot development of selected elements
plt.pcolor(A[:,1:100], label = "observed")
plt.xlabel("iteration")
plt.ylabel("vector element")
plt.title("CombUCB1 exploration (first 100 iterations)")
yellow_patch = mpatches.Patch(color='yellow', label='observed')
plt.legend(handles=[yellow_patch], bbox_to_anchor=(0.73, 0.13), loc=2)
plt.savefig('basicCombUCB1_100iters.pdf')

plt.pcolor(A[:,100:200], label = "observed")
plt.xlabel("iteration")
plt.ylabel("vector element")
plt.title("CombUCB1 exploration (iterations 101 to 200)")
yellow_patch = mpatches.Patch(color='yellow', label='observed')
plt.legend(handles=[yellow_patch], bbox_to_anchor=(0.73, 0.13), loc=2)
plt.savefig('basicCombUCB1_100_200iters.pdf')

plt.pcolor(A[:,4000:4100], label = "observed")
plt.xlabel("iteration")
plt.ylabel("vector element")
plt.title("CombUCB1 exploration (iterations 4001 to 4100)")
yellow_patch = mpatches.Patch(color='yellow', label='observed')
plt.legend(handles=[yellow_patch], bbox_to_anchor=(0.73, 0.13), loc=2)
plt.savefig('basicCombUCB1_4000_4100iters.pdf')


# Calculate the regret based on the weights observed vs weights of optimal action
# Start from column index 1 as 0 is the initialisation
actual_return = np.sum(w_observed[:,1:])
best_return = np.sum(w[-K:,1:])
regret = best_return-actual_return






### FPL-TRiX
# model params
eta = 0.1 # this acts against the perturbations, so the higher this is the less significant the perturbations in the combinatorial optimisation
gamma = 10.0 # the higher this is the less the alorithm explores
Bt = 0.01 # higher B_t means more exploration

# set matrices for storing results
Z = np.zeros((L, iters))
A = np.zeros((L, iters))
l_hat = np.zeros((L, iters))
L_hat = np.zeros((L, iters))

### Run algorithm from t=1,,, (start at 1 as L_hat is initialised as 0s, so 0th column is this init)
for t in range(iters)[1:]:
    
    # draw perturbation vector
    Z[:,t] = truncexp.rvs(Bt, L)
    
    # select best action to minimise expected cost
    A[np.argsort(eta*L_hat[:,(t-1)] - Z[:,t])[-K:],t] = 1
        
    # calculate l_hats
    l_hat[:,t] = w[:,t]/(np.sum(A, axis=1)/t + gamma)
    
    # update L_hats
    L_hat[:,t] = L_hat[:,(t-1)] + l_hat[:,t]

FPL_actual_return = np.sum(w[:,1:]*A[:,1:])

# plot development of selected elements
plt.pcolor(A[:,1:100], label = "observed")
plt.xlabel("iteration")
plt.ylabel("vector element")
plt.title("FPL_TRiX exploration (first 100 iterations)")
yellow_patch = mpatches.Patch(color='yellow', label='observed')
plt.legend(handles=[yellow_patch], bbox_to_anchor=(0.73, 0.13), loc=2)
plt.savefig('basicFPL_100iters.pdf')

plt.pcolor(A[:,100:200], label = "observed")
plt.xlabel("iteration")
plt.ylabel("vector element")
plt.title("FPL-TRiX exploration (iterations 101 to 200)")
yellow_patch = mpatches.Patch(color='yellow', label='observed')
plt.legend(handles=[yellow_patch], bbox_to_anchor=(0.73, 0.13), loc=2)
plt.savefig('basicFPL_100_200iters.pdf')

plt.pcolor(A[:,4000:4100], label = "observed")
plt.xlabel("iteration")
plt.ylabel("vector element")
plt.title("FPL-TRiX exploration (iterations 4001 to 4100)")
yellow_patch = mpatches.Patch(color='yellow', label='observed')
plt.legend(handles=[yellow_patch], bbox_to_anchor=(0.73, 0.13), loc=2)
plt.savefig('basicFPL_4000_4100iters.pdf')


FPL_actual_return = np.sum(w[:,1:]*A[:,1:])
FPL_best_return = np.sum(w[-K:,1:])
FPL_regret = FPL_best_return-FPL_actual_return

np.sum(w[1:4,1:])




