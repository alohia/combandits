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

# IDEA:
### Could Bt parameter be determined with an algorithm that adjusts it over time, perhaps if costs increase/change?

################################################
### impurity function to measure exploration ###
################################################
def impurity(path_matrix):
    row_sums = np.sum(path_matrix, axis=1)
    pc_chosen = row_sums/np.shape(path_matrix)[1]
    row_impurities = pc_chosen*(1-pc_chosen)*4
    average_impurity = np.mean(row_impurities)
    impurity_score = average_impurity*1000
    return impurity_score


###############################
### setup FPL-TRiX function ###
###############################

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

# draw weights for all iterations
w = np.zeros((L, iters))
for t in range(iters):
    # select path means
    path_means = path_means2

    # draw weights
    w[:,t] = np.concatenate(truncnorm.rvs((lower-path_means)/sigma, (upper-path_means)/sigma, loc=path_means, scale=sigma)*diag_zero)


###############################
### setup FPL-TRiX function ###
###############################

def fpl_trix(eta, gamma, Bt):        
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
        
        # calculate l_hats
        l_hat[:,t] = w_observed[:,t]/(np.sum(A, axis=1)/t + gamma)
        
        # update L_hats
        L_hat[:,t] = L_hat[:,(t-1)] + l_hat[:,t]

    cost = stochastic_network_FPL_cost = np.sum(A*w)
    exploration_score = impurity(A)
    return exploration_score, cost

test = fpl_trix(5, 0.001, 5)
###############################################################################
### changing params and storing and plotting the exploration score and cost ###
###############################################################################
# parameter ranges to iterate over
etas = list((0.001, 0.1, 1, 5, 10, 25, 50, 75, 100))
gammas = list((0.1, 0.05, 0.01, 0.0001, 0.000001))
Bts = list((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

# set constants to use when varying the other parameters
eta = 5
gamma = 0.001
Bt = 5

# changing etas
etas_exploration_score = np.zeros(len(etas))
etas_cost = np.zeros(len(etas))
for e in range(len(etas)):
    temp_store = fpl_trix(etas[e], gamma, Bt)
    etas_exploration_score[e] = temp_store[0]
    etas_cost[e] = temp_store[1]

#etas_exploration_score_store = etas_exploration_score[:]
#etas_cost_store = etas_cost[:]
etas_exploration_score_store = np.vstack((etas_exploration_score_store, etas_exploration_score))
etas_cost_store = np.vstack((etas_cost_store, etas_cost))

# changing gammas
gammas_exploration_score = np.zeros(len(gammas))
gammas_cost = np.zeros(len(gammas))
for g in range(len(gammas)):
    temp_store = fpl_trix(eta, gammas[g], Bt)
    gammas_exploration_score[g] = temp_store[0]
    gammas_cost[g] = temp_store[1]

#gammas_exploration_score_store = gammas_exploration_score[:]
#gammas_cost_store = gammas_cost[:]
gammas_exploration_score_store = np.vstack((gammas_exploration_score_store, gammas_exploration_score))
gammas_cost_store = np.vstack((gammas_cost_store, gammas_cost))

# changing Bts
Bts_exploration_score = np.zeros(len(Bts))
Bts_cost = np.zeros(len(Bts))
for b in range(len(Bts)):
    temp_store = fpl_trix(eta, gamma, Bts[b])
    Bts_exploration_score[b] = temp_store[0]
    Bts_cost[b] = temp_store[1]

#Bts_exploration_score_store = Bts_exploration_score[:]
#Bts_cost_store = Bts_cost[:]
Bts_exploration_score_store = np.vstack((Bts_exploration_score_store, Bts_exploration_score))
Bts_cost_store = np.vstack((Bts_cost_store, Bts_cost))

# plot eta exploration score and cost
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(etas, np.mean(etas_exploration_score_store, axis=0), color=(0.0, 0.2, 0.9))
ax2.plot(etas, np.mean(etas_cost_store, axis=0), color=(0.9, 0.0, 0.5))
ax1.set_xlabel('eta')
ax1.set_ylabel('exploration score', color=(0.0, 0.2, 0.9))
ax2.set_ylabel('total cost', color=(0.9, 0.0, 0.5))
plt.savefig('eta_expl_cost.pdf')

# plot gamma exploration score and cost
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(gammas, np.mean(gammas_exploration_score_store, axis=0), color=(0.0, 0.2, 0.9))
ax2.plot(gammas, np.mean(gammas_cost_store, axis=0), color=(0.9, 0.0, 0.5))
ax1.set_xlabel('gamma')
ax1.set_ylabel('exploration score', color=(0.0, 0.2, 0.9))
ax2.set_ylabel('total cost', color=(0.9, 0.0, 0.5))
plt.savefig('gamma_expl_cost.pdf')

# plot Bt exploration score and cost
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(Bts, np.mean(Bts_exploration_score_store, axis=0), color=(0.0, 0.2, 0.9))
ax2.plot(Bts, np.mean(Bts_cost_store, axis=0), color=(0.9, 0.0, 0.5))
ax1.set_xlabel('B_t')
ax1.set_ylabel('exploration score', color=(0.0, 0.2, 0.9))
ax2.set_ylabel('total cost', color=(0.9, 0.0, 0.5))
plt.savefig('Bt_expl_cost.pdf')


# plot chosen paths by FPL-TRiX
plt.pcolor(A_FPL[:,1:1000], label = "edges in chosen path")
plt.xlabel("iteration")
plt.ylabel("path location in network matrix")
plt.title("FPL_TRiX chosen path at each iteration")
yellow_patch = mpatches.Patch(color='yellow', label='edges in chosen path')
plt.legend(handles=[yellow_patch], bbox_to_anchor=(0.53, 0.98), loc=2)
plt.savefig('stoch_network_FPL.pdf')

