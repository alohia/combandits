#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:11:38 2017

@author: angus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
t = 0
# Initial draw of weight for each element
w[:,t] = np.random.binomial(1, bern_means)
w_observed[:,t] = w[:,t]

# counter for number of observations of each element
c = np.array([1]*L)

### run algorithm over iterations 1 to number of iterations
for t in list(range(iters))[1:]:
    # Calculate upper confidence bounds
    U[:,t] = np.sum(w_observed, axis=1)/c + np.sqrt(1.5*np.log(t)/c)
    
    # Select 8 highest upper confidence bounds to play next round
    A[np.argsort(U[:,t])[-K:],t] = 1
    
    # Draw new weights and store in w
    w[:,t] = np.random.binomial(1, bern_means)
    
    # Store observed weights (0 otherwise) and add 1 to the counts
    w_observed[:,t] = w[:,t]*A[:,t]
    c[A[:,t]==1] += 1

# plot development of selected elements
plt.pcolor(A[1:])

# Calculate the regret based on the weights observed vs weights of optimal action
# Start from column index 1 as 0 is the initialisation
actual_return = np.sum(w_observed[:,1:])
best_return = np.sum(w[-K:,1:])
regret = best_return-actual_return

# Calculate upper bounds
delta_e_min = bern_means.iloc[-K] - bern_means[:-K]

upper_bound1 = sum(K*534*np.log(iters)/delta_e_min) + (np.pi**2/3+1)*K*L

upper_bound2 = 47*np.sqrt(K*L*iters*np.log(iters)) + (np.pi**2/3+1)*K*L
lower_bound2 = min(np.sqrt(K*L*iters), K*iters)/20

### Creating function to test out different parameters
def combUCB1(K, L, iterations):
    # Set arrays to store at each iteration
    U = np.zeros((L, iterations))
    A = np.zeros((L, iterations))
    w = np.zeros((L, iterations))
    w_observed = np.zeros((L, iterations))
    
    # set means of each element
    bern_means = pd.Series(list(range(L))).map(lambda x: x/L)
    
    # INIT
    t = 0
    # Initial draw of weight for each element
    w[:,t] = np.random.binomial(1, bern_means)
    w_observed[:,t] = w[:,t]
    
    # counter for number of observations of each element
    c = np.array([1]*L)
    
    # run algorithm over iterations 1 to number of iterations
    for t in list(range(iterations))[1:]:
        # Calculate upper confidence bounds
        U[:,t] = np.sum(w_observed, axis=1)/c + np.sqrt(1.5*np.log(t)/c)
        
        # Select 8 highest upper confidence bounds to play next round
        A[np.argsort(U[:,t])[-K:],t] = 1
        
        # Draw new weights and store in w
        w[:,t] = np.random.binomial(1, bern_means)
        
        # Store observed weights (0 otherwise) and add 1 to the counts
        w_observed[:,t] = w[:,t]*A[:,t]
        c[A[:,t]==1] += 1
    
    # Calcualte the regret based on the weights observed vs weights of optimal action
    # Start from column index 1 as 0 is the initialisation
    actual_return = np.sum(w_observed[:,1:])
    best_return = np.sum(w[-K:,1:])
    regret = best_return-actual_return
    
    return(regret)


### Plot regret as a function of L (should be linear)
ls = list(range(100, 1001, 10))
regrets_l = ls[:]
for l in range(len(ls)):
    regrets_l[l] = combUCB1(K=20, L=ls[l], iterations=1000)

plt.scatter(x=ls, y=regrets_l)
plt.title('regret as function of cardinality of ground set (L)')
plt.xlabel('L')
plt.ylabel('regret')
plt.show()
plt.close()


### Plot regret as a function of K (should be linear)
ks = list(range(1,101))
regrets_k = ks[:]
for k in range(len(ks)):
    regrets_k[k] = combUCB1(K=ks[k], L=1000, iterations=1000)

plt.scatter(x=ks, y=regrets_k)
plt.title('regret as function of number of chosen items (K)')
plt.xlabel('K')
plt.ylabel('regret')
plt.show()
plt.close()


### Plot regret as a function of K
iters = list(range(100, 1001, 100))
regrets_i = iters[:]
for i in range(len(iters)):
    regrets_i[i] = combUCB1(K=20, L=1000, iterations=iters[i])

plt.scatter(x=iters, y=regrets_i)
plt.title('regret as function of number of iterations/time')
plt.xlabel('time')
plt.ylabel('regret')
plt.show()
plt.close()

















