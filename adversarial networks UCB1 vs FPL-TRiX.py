#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 13:48:19 2017

@author: angus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
truncnorm = stats.truncnorm
truncexp = stats.truncexpon
from tqdm import tqdm

### Set parameters
V = 10
L = V*V
iters = 10000

### IDEAS
### Could Bt parameter be determined with an algorithm that adjusts it over time, perhaps if costs increase/change significantly?

### TO CHECK/DISCUSS
# Does UCB1 work more or less the same when subtracting the bound adjustment instead of adding (i.e. more like LCB)?
# For FPL does adding the random perturbations instead of subtracting them work the same?

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
sigma = 0.01

### arrays to store results form each iteration
U = np.zeros((L, iters))
A = np.zeros((L, iters))
w = np.zeros((L, iters))
w_observed = np.zeros((L, iters))

exploration_parameter = 1.5

### INIT
# draw weights for all iterations
# Draw new weights and store in w
for t in range(iters):
    # select path means
    if t < iters/2:
        path_means = path_means1
    else:
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

# calculature total cost
adversarial_network_UCB_paths = A[:]
adversrial_network_UCB_cost = np.sum(adversarial_network_UCB_paths*w)
adversrial_network_UCB_exploration_score = impurity(adversarial_network_UCB_paths)



### find shortest paths for each iteration
# calculate actual means
actual_path_means1 = truncnorm.stats(np.concatenate(lower-path_means1)/sigma, np.concatenate(upper-path_means1)/sigma, loc=np.concatenate(path_means1), scale=sigma, moments='m').reshape(V, V)
actual_path_means2 = truncnorm.stats(np.concatenate(lower-path_means2)/sigma, np.concatenate(upper-path_means2)/sigma, loc=np.concatenate(path_means2), scale=sigma, moments='m').reshape(V, V)
adversarial_network_optimal_paths = np.zeros((L, iters))

for t in tqdm(range(iters)):
    if t < iters/2:
        path_means = actual_path_means1
    else:
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
            if distances[u] > distances[v] + path_means[v, u]:
                distances[u] = distances[v] + path_means[v, u]
                paths[u] = paths[v][:]
                paths[u].append(u)
                
    shortest_path = paths[-1]
    shortest_path_matrix = np.zeros((V, V))
    for i in range(len(shortest_path)-1):
        shortest_path_matrix[shortest_path[i], shortest_path[i+1]] = 1
    adversarial_network_optimal_paths[:,t] = np.concatenate(shortest_path_matrix)

# calculate total costs
adversrial_network_optimal_path_cost = np.sum(adversarial_network_optimal_paths*w)



### FPL-TRiX on same adversarial network
# model params
eta = 2 # this acts against the perturbations, so the higher this is the less significant the perturbations in the combinatorial optimisation
gamma = 0.0001 # the lower this is the more the alorithm explores
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

adversarial_network_FPL_paths = A[:]
adversrial_network_FPL_cost = np.sum(adversarial_network_FPL_paths*w)
adversrial_network_FPL_exploration_score = impurity(adversarial_network_FPL_paths)

plt.pcolormesh(adversarial_network_optimal_paths)
plt.pcolormesh(adversarial_network_FPL_paths)
plt.pcolormesh(adversarial_network_UCB_paths)


# first 1000 iters
np.sum(adversarial_network_optimal_paths[:,:1000]*w[:,:1000])
np.sum(adversarial_network_UCB_paths[:,:1000]*w[:,:1000])
np.sum(adversarial_network_FPL_paths[:,:1000]*w[:,:1000])
# middle 1000
np.sum(adversarial_network_optimal_paths[:,4500:5500]*w[:,4500:5500])
np.sum(adversarial_network_UCB_paths[:,4500:5500]*w[:,4500:5500])
np.sum(adversarial_network_FPL_paths[:,4500:5500]*w[:,4500:5500])
# last 1000
np.sum(adversarial_network_optimal_paths[:,9000:]*w[:,9000:])
np.sum(adversarial_network_UCB_paths[:,9000:]*w[:,9000:])
np.sum(adversarial_network_FPL_paths[:,9000:]*w[:,9000:])






##########################################
### setup network where UCB gets stuck ###
##########################################

diag_zero = np.ones(L).reshape(V, V) - np.diag(np.ones(V))

path_means3 = np.ones(L).reshape(V, V)
for i in range(V):
    for j in range(V):
        path_means3[i,j] = abs(i - j)/V + 5/(V*2)
path_means3 = abs(path_means3 * diag_zero)


path_means4 = np.ones(L).reshape(V, V)
for i in range(V-1):
    path_means4[i, i+1] = 1/(V*2)




# trunc norm params
lower = 0
upper = 1
sigma = 0.01

### arrays to store results form each iteration
U = np.zeros((L, iters))
A = np.zeros((L, iters))
w = np.zeros((L, iters))
w_observed = np.zeros((L, iters))

exploration_parameter = 1.5


### INIT
# draw weights for all iterations
# Draw new weights and store in w
for t in range(iters):
    # select path means
    if int(t/1000)/2==int(int(t/1000)/2):
        path_means = path_means3
    else:
        path_means = path_means4

    # draw weights
    w[:,t] = np.concatenate(truncnorm.rvs((lower-path_means)/sigma, (upper-path_means)/sigma, loc=path_means, scale=sigma)*diag_zero)


t = 0
# Initial obseration of all weight for each element
w_observed[:,t] = w[:,t]

# counter for number of observations of each element
c = np.array([1]*L)


### run algorithm over iterations 1 to number of iterations
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

# calculate total costs
adversarial_network2_UCB_paths = A[:]
adversrial_network2_UCB_cost = np.sum(adversarial_network2_UCB_paths*w)
adversrial_network2_UCB_exploration_score = impurity(adversarial_network2_UCB_paths)


### find shortest paths for each iteration
# calculate actual means
actual_path_means3 = truncnorm.stats(np.concatenate(lower-path_means3)/sigma, np.concatenate(upper-path_means3)/sigma, loc=np.concatenate(path_means3), scale=sigma, moments='m').reshape(V, V)
actual_path_means4 = truncnorm.stats(np.concatenate(lower-path_means4)/sigma, np.concatenate(upper-path_means4)/sigma, loc=np.concatenate(path_means4), scale=sigma, moments='m').reshape(V, V)

adversarial_network2_optimal_paths = np.zeros((L, iters))

for t in tqdm(range(iters)):
    if int(t/1000)/2==int(int(t/1000)/2):
        path_means = path_means3
    else:
        path_means = path_means4
    
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
            if distances[u] > distances[v] + path_means[v, u]:
                distances[u] = distances[v] + path_means[v, u]
                paths[u] = paths[v][:]
                paths[u].append(u)
                
    shortest_path = paths[-1]
    shortest_path_matrix = np.zeros((V, V))
    for i in range(len(shortest_path)-1):
        shortest_path_matrix[shortest_path[i], shortest_path[i+1]] = 1
    adversarial_network2_optimal_paths[:,t] = np.concatenate(shortest_path_matrix)

adversrial_network2_optimal_path_cost = np.sum(adversarial_network2_optimal_paths*w)


### FPL-TRiX on same adversarial network
# model params
eta = 0.1 # this acts against the perturbations, so the higher this is the less significant the perturbations in the combinatorial optimisation
gamma = 0.001 # the higher this is the more the alorithm explores
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

adversarial_network2_FPL_paths = A[:]
adversrial_network2_FPL_cost = np.sum(adversarial_network2_FPL_paths*w)
adversrial_network2_FPL_exploration_score = impurity(adversarial_network2_FPL_paths)

plt.pcolormesh(adversarial_network2_optimal_paths)
plt.pcolormesh(adversarial_network2_UCB_paths[:,:])
plt.pcolormesh(adversarial_network2_FPL_paths[:,:])


# first 1000 iters
np.sum(adversarial_network2_optimal_paths[:,:1000]*w[:,:1000])
np.sum(adversarial_network2_UCB_paths[:,:1000]*w[:,:1000])
np.sum(adversarial_network2_FPL_paths[:,:1000]*w[:,:1000])
# middle 1000
np.sum(adversarial_network2_optimal_paths[:,4500:5500]*w[:,4500:5500])
np.sum(adversarial_network2_UCB_paths[:,4500:5500]*w[:,4500:5500])
np.sum(adversarial_network2_FPL_paths[:,4500:5500]*w[:,4500:5500])
# last 1000
np.sum(adversarial_network2_optimal_paths[:,9000:]*w[:,9000:])
np.sum(adversarial_network2_UCB_paths[:,9000:]*w[:,9000:])
np.sum(adversarial_network2_FPL_paths[:,9000:]*w[:,9000:])

# last 1000
np.sum(adversarial_network2_optimal_paths[:,8000:9000]*w[:,8000:9000])
np.sum(adversarial_network2_UCB_paths[:,8000:9000]*w[:,8000:9000])
np.sum(adversarial_network2_FPL_paths[:,8000:9000]*w[:,8000:9000])



### plot increase in cost over time
adversarial_network2_UCB_cumul_cost = np.zeros(iters/10)
adversarial_network2_FPL_cumul_cost = np.zeros(iters/10)
for t in tqdm(range(0, iters, 10)):
    adversarial_network2_UCB_cumul_cost[t/10] = np.sum(adversarial_network2_UCB_paths[:,:t]*w[:,:t])
    adversarial_network2_FPL_cumul_cost[t/10] = np.sum(adversarial_network2_FPL_paths[:,:t]*w[:,:t])

x_axis = list(range(0, iters, 10))

plt.plot(x_axis, adversarial_network2_UCB_cumul_cost, linewidth = 2.5, c = (1, 0.5, 0.0), alpha = 1, label = "CombUCB1")
plt.plot(x_axis, adversarial_network2_FPL_cumul_cost, linewidth = 2.5, c = (0.0, 0.5, 1), alpha = 1, label = "FPL-TRiX")
plt.xlabel("iteration")
plt.ylabel("cumulative cost")
plt.title("Adversarial network 2 cumulative cost")
plt.legend(ncol = 2)
plt.savefig('advers_net2_cumul_costs.jpg')




###############################################################
### setup network that changes each iteration with sin wave ###
###############################################################
iters = 10000
# setup different 'shifts' to the sine wave so that different paths have different starting points on the wave
sine_wave_shift = np.zeros(L).reshape(V, V)
for i in range(V):
    for j in range(V):
        sine_wave_shift[i,j] = abs(i - j)/(V)

# parameter which determines how many repetitions of the sine wave
sine_wave_reps = 2

# set an adjustment to subtract from each path weight mean to encourage longer paths
adjust = 0.00

# setting trunc norm means for each iteration
trunc_norm_means = np.zeros((L, iters))
for t in range(iters):
    trunc_norm_means[:,t] = np.concatenate(((np.sin((sine_wave_shift+t*sine_wave_reps/(iters))*np.pi)+1)/2-adjust)*diag_zero)

# draw weights for entire period
# trunc norm params
lower = 0
upper = 1
sigma = 0.1

w = truncnorm.rvs((lower-trunc_norm_means)/sigma, (upper-trunc_norm_means)/sigma, loc=trunc_norm_means, scale=sigma)


### arrays to store results form each iteration
U = np.zeros((L, iters))
A = np.zeros((L, iters))
w_observed = np.zeros((L, iters))

exploration_parameter = 1.5


### INIT

t = 0
# Initial draw of weight for each element
w_observed[:,t] = w[:,t]

# counter for number of observations of each element
c = np.array([1]*L)


### run algorithm over iterations 1 to number of iterations
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

# calculate total costs
sine_adversrial_network_UCB_paths = A[:]
sine_adversrial_network_UCB_cost = np.sum(sine_adversrial_network_UCB_paths*w)
sine_adversrial_network_UCB_exploration_score = impurity(sine_adversrial_network_UCB_paths)


### find shortest paths for each iteration
# find actual means
actual_trunc_norm_means = truncnorm.stats(np.concatenate(lower-trunc_norm_means)/sigma, np.concatenate(upper-trunc_norm_means)/sigma, loc=np.concatenate(trunc_norm_means), scale=sigma, moments='m').reshape(L, iters)

sine_adversary_optimal_paths = np.zeros((L, iters))

for t in tqdm(range(iters)):
    path_matrix = actual_trunc_norm_means[:,t].reshape(V,V)
    
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
            if distances[u] > distances[v] + path_matrix[v, u]:
                distances[u] = distances[v] + path_matrix[v, u]
                paths[u] = paths[v][:]
                paths[u].append(u)
                
    shortest_path = paths[-1]
    shortest_path_matrix = np.zeros((V, V))
    for i in range(len(shortest_path)-1):
        shortest_path_matrix[shortest_path[i], shortest_path[i+1]] = 1
    sine_adversary_optimal_paths[:,t] = np.concatenate(shortest_path_matrix)



sine_adversrial_network_optimal_path_cost = np.sum(sine_adversary_optimal_paths*w)

sine_UCB_end_cost = np.sum((A*w)[:,70000:])
sine_optimal_path_end_cost = np.sum((sine_adversary_optimal_paths*w)[:,70000:])

### FPL-TRiX on same adversarial network
# model params
eta = 1 # this acts against the perturbations, so the higher this is the less significant the perturbations in the combinatorial optimisation
gamma = 0.1 # the higher this is the more the alorithm explores
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

sine_adversrial_network_FPL_paths = A[:]
sine_adversrial_network_FPL_cost = np.sum(sine_adversrial_network_FPL_paths*w)
sine_adversrial_network_FPL_exploration_score = impurity(sine_adversrial_network_FPL_paths)


plt.pcolormesh(sine_adversrial_network_FPL_paths[:,:10000])
plt.pcolormesh(sine_adversrial_network_UCB_paths[:,:10000])
plt.pcolormesh(sine_adversary_optimal_paths[:,:10000])

# first 1000 iters
np.sum(sine_adversary_optimal_paths[:,:1000]*w[:,:1000])
np.sum(sine_adversrial_network_UCB_paths[:,:1000]*w[:,:1000])
np.sum(sine_adversrial_network_FPL_paths[:,:1000]*w[:,:1000])
# middle 1000
np.sum(sine_adversary_optimal_paths[:,4500:5500]*w[:,4500:5500])
np.sum(sine_adversrial_network_UCB_paths[:,4500:5500]*w[:,4500:5500])
np.sum(sine_adversrial_network_FPL_paths[:,4500:5500]*w[:,4500:5500])
# last 1000
np.sum(sine_adversary_optimal_paths[:,9000:]*w[:,9000:])
np.sum(sine_adversrial_network_UCB_paths[:,9000:]*w[:,9000:])
np.sum(sine_adversrial_network_FPL_paths[:,9000:]*w[:,9000:])



### LOOK AT EXPLORATION OVER TIME



