#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:37:29 2017

@author: angus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
truncnorm = stats.truncnorm
truncexp = stats.truncexpon
from tqdm import tqdm

### Set parameters
V = 10
L = V*V
iters = 10000


### IDEAS
# Show that UCB1 explored and finds the optimal paths
# But should continue to explore a little bit
# If things keep changing a bit then it should keep exploring
# But if it gets very confident with one setting and then things change then it may not adapt well
# So show 1. setting that makes it confident and 2. timeframe that makes it confident
# show costs of UCB1 vs costs of FPL-TRiX, and when one is higher than the other etc

### Could Bt parameter be determined with an algorithm that adjusts it over time, perhaps if costs increase/change?


### TO CHECK/DISCUSS
# Does UCB1 work more or less the same by subtracting the bound adjustment instead of adding (i.e. more like LCB)
# For FPL does adding the random perturbations instead of subtracting them work the same?


##################################################################
### setup network which changes after a certain length of time ###
##################################################################

### set mean of each path
diag_zero = np.ones(L).reshape(V, V) - np.diag(np.ones(V))

path_means1 = np.zeros(L).reshape(V, V)
for i in range(V):
    for j in range(V):
        path_means1[i,j] = abs(i - j)/V - 1/(V*2)
path_means1 = abs(path_means1 * diag_zero)

path_means2 = np.random.uniform(0, 1, L).reshape(V, V) * diag_zero
path_means2[0,-1] = 1 # to try to make sure best path isn't to go straight to the end!

# trunc norm params
lower = 0
upper = 1
sigma = 0.1

### arrays to store results form each iteration
U = np.zeros((L, iters))
A = np.zeros((L, iters))
w = np.zeros((L, iters))
w_observed = np.zeros((L, iters))

exploration_parameter = 1.5


### INIT
# draw weights for all iterations
for t in range(iters):
    # select path means
    path_means = path_means2

    # draw weights
    w[:,t] = np.concatenate(truncnorm.rvs((lower-path_means)/sigma, (upper-path_means)/sigma, loc=path_means, scale=sigma)*diag_zero)


t = 0
# initial ovservation of each weight
w_observed[:,t] = w[:,t]

# counter for number of observations of each element
c = np.array([1]*L)


### run UCB algorithm over iterations 1 to number of iterations
for t in tqdm(range(1, iters)):
    # Calculate lower confidence bounds
    U[:,t] = np.maximum(0, np.sum(w_observed, axis=1)/c - np.sqrt(exploration_parameter*np.log(t)/c))
    U[:,t] = np.concatenate(U[:,t].reshape(V,V) * diag_zero)
    
    ### Select shortest path based on the upper confidence bounds
    U_matrix_form = U[:,t].reshape(V,V)
    # set initial conditions
    unvisited = list(range(V))
    distances = [999999]*V
    distances[0] = 0
    paths = [[0]]*V
    # find shortest path
    while len(unvisited) > 0:
        i = np.argmin([distances[i] for i in unvisited])
        v = unvisited[i]
        unvisited.pop(i)
        for u in range(V):
            if distances[u] > distances[v] + U_matrix_form[v, u]:
                distances[u] = distances[v] + U_matrix_form[v, u]
                paths[u] = paths[v][:]
                paths[u].append(u)
                
    shortest_path = paths[-1]
    
    # store observed paths
    observed_matrix = A[:,t].reshape(V,V)
    for i in range(len(shortest_path)-1):
        observed_matrix[shortest_path[i], shortest_path[i+1]] = 1
    A[:,t] = np.concatenate(observed_matrix)
    
    # Store observed weights (0 otherwise) and add 1 to the counts
    w_observed[:,t] = w[:,t]*A[:,t]
    c[A[:,t]==1] += 1

A_UCB = A[:,:]

### find shortest paths for each iteration
# calculate actual means
actual_path_means1 = truncnorm.stats(np.concatenate(lower-path_means1)/sigma, np.concatenate(upper-path_means1)/sigma, loc=np.concatenate(path_means1), scale=sigma, moments='m').reshape(V, V)
actual_path_means2 = truncnorm.stats(np.concatenate(lower-path_means2)/sigma, np.concatenate(upper-path_means2)/sigma, loc=np.concatenate(path_means2), scale=sigma, moments='m').reshape(V, V)

stochastic_network_optimal_paths = np.zeros((L, iters))

for t in tqdm(range(iters)):
    path_means = actual_path_means2
    
    unvisited = list(range(V))
    distances = [999999]*V
    distances[0] = 0
    paths = [[0]]*V
    # find shortest path
    while len(unvisited) > 0:
        i = np.argmin([distances[i] for i in unvisited])
        v = unvisited[i]
        unvisited.pop(i)
        for u in range(V):
            if distances[u] > distances[v] + path_means2[v, u]:
                distances[u] = distances[v] + path_means[v, u]
                paths[u] = paths[v][:]
                paths[u].append(u)
                
    shortest_path = paths[-1]
    shortest_path_matrix = np.zeros((V, V))
    for i in range(len(shortest_path)-1):
        shortest_path_matrix[shortest_path[i], shortest_path[i+1]] = 1
    stochastic_network_optimal_paths[:,t] = np.concatenate(shortest_path_matrix)

# calculate total costs
stochastic_network_UCB_cost = np.sum(A_UCB*w)
stochastic_network_optimal_path_cost = np.sum(stochastic_network_optimal_paths*w)

# optimal path
plt.pcolor(stochastic_network_optimal_paths[:,1:1000], label = "edges in optimal path")
plt.xlabel("iteration")
plt.ylabel("path location in network matrix")
plt.title("optimal path to take at each iteration")
yellow_patch = mpatches.Patch(color='yellow', label='edges in optimal path')
plt.legend(handles=[yellow_patch], bbox_to_anchor=(0.53, 0.98), loc=2)
plt.savefig('stoch_network_opt_paths.pdf')

# plot chosen paths by CombUCB1
plt.pcolor(A_UCB[:,1:1000], label = "edges in chosen path")
plt.xlabel("iteration")
plt.ylabel("path location in network matrix")
plt.title("CombUCB1 chosen path at each iteration")
yellow_patch = mpatches.Patch(color='yellow', label='edges in chosen path')
plt.legend(handles=[yellow_patch], bbox_to_anchor=(0.53, 0.98), loc=2)
plt.savefig('stoch_network_UCB1.pdf')



### FPL-TRiX on same adversarial network
# model params
eta = 1 # this acts against the perturbations, so the higher this is the less significant the perturbations in the combinatorial optimisation
gamma = 0.01 # the lower this is the more the alorithm explores
Bt = 0.5 # controls the perturbations, increasing this should increase exploration

# matrices to store results
Z = np.zeros((L, iters))
A = np.zeros((L, iters))
w_observed = np.zeros((L, iters))
l_hat = np.zeros((L, iters))
L_hat = np.zeros((L, iters))


### Run algorithm from t=1,,, (start at 1 as L_hat is initialised as 0s, so 0th column is this init)
for t in tqdm(range(1, iters)):
    
    # draw perturbation vector
    Z[:,t] = truncexp.rvs(Bt, size=L)
    
    # create path matrix of perturbed leaders
    perturbed_leader_matrix = (eta*L_hat[:,(t-1)] + Z[:,t]).reshape(V, V)
    
    unvisited = list(range(V))
    distances = [999999]*V
    distances[0] = 0
    paths = [[0]]*V
    # find shortest path
    while len(unvisited) > 0:
        i = np.argmin([distances[i] for i in unvisited])
        v = unvisited[i]
        unvisited.pop(i)
        for u in range(V):
            if distances[u] > distances[v] + perturbed_leader_matrix[v, u]:
                distances[u] = distances[v] + perturbed_leader_matrix[v, u]
                paths[u] = paths[v][:]
                paths[u].append(u)
                
    shortest_path = paths[-1]
    
    # store observed paths
    observed_matrix = A[:,t].reshape(V,V)
    for i in range(len(shortest_path)-1):
        observed_matrix[shortest_path[i], shortest_path[i+1]] = 1
    A[:,t] = np.concatenate(observed_matrix)
    
    # Store observed weights (0 otherwise) and add 1 to the counts
    w_observed[:,t] = w[:,t]*A[:,t]
    c[A[:,t]==1] += 1
    
    # calculate l_hats
    l_hat[:,t] = w_observed[:,t]/(np.sum(A, axis=1)/t + gamma)
    
    # update L_hats
    L_hat[:,t] = L_hat[:,(t-1)] + l_hat[:,t]

A_FPL = A[:,:]

stochastic_network_FPL_cost = np.sum(A*w)


# plot chosen paths by FPL-TRiX
plt.pcolor(A_FPL[:,1:1000], label = "edges in chosen path")
plt.xlabel("iteration")
plt.ylabel("path location in network matrix")
plt.title("FPL_TRiX chosen path at each iteration")
yellow_patch = mpatches.Patch(color='yellow', label='edges in chosen path')
plt.legend(handles=[yellow_patch], bbox_to_anchor=(0.53, 0.98), loc=2)
plt.savefig('stoch_network_FPL.pdf')

