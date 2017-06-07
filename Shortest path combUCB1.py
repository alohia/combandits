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

# Scenario: User picks a path from vertex 0 to vertex 14
# At each play the cost of each edge of the path is drawn from a different distribution for each edge
# Algorithm attempts to learn the best path to take to minimise expected cost

### Set parameters
V = 5
L = V*V
iters = 10000

### Set arrays to store at each iteration
# an upper confidence bound for each edge (U)
# the path selected to play in the next round (A)
# costs/weights drawn for each edge at each round (w)
# observed costs/weights for the chosen path (w_observed)

U = np.zeros((L, iters))
A = np.zeros((L, iters))
w = np.zeros((L, iters))
w_observed = np.zeros((L, iters))

### set means of each path
path_maxes = np.random.uniform(0, 1, L).reshape(V, V)
path_maxes = path_maxes - np.diag(np.diag(path_maxes))
path_maxes[0,-1] = 1 # to ensure best path isn't to go straight to the end!

### INIT
t = 0
# Initial draw of weight for each element
w[:,t] = np.random.uniform(0, np.concatenate(path_maxes))
w_observed[:,t] = w[:,t]

# counter for number of observations of each element
c = np.array([1]*L)


### run algorithm over iterations 1 to number of iterations
for t in list(range(iters))[1:]:
    # Calculate lower confidence bounds
    U[:,t] = np.maximum(0, np.sum(w_observed, axis=1)/c - np.sqrt(1.5*np.log(t)/c))
    U[:,t] = np.concatenate(U[:,t].reshape(V,V) - np.diag(np.diag(U[:,t].reshape(V,V))))
    
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
    
    # Draw new weights and store in w
    w[:,t] = np.random.uniform(0, np.concatenate(path_maxes))
    
    # Store observed weights (0 otherwise) and add 1 to the counts
    w_observed[:,t] = w[:,t]*A[:,t]
    c[A[:,t]==1] += 1


# Find shortest path from path_maxes matrix to work out optimal path
unvisited = list(range(V))
distances = [999999]*V
distances[0] = 0
optimal_paths = [[0]]*V

while len(unvisited) > 0:
    i = np.argmin([distances[i] for i in unvisited])
    v = unvisited[i]
    unvisited.pop(i)
    for u in range(V):
        if distances[u] > distances[v] + path_maxes[v, u]:
            distances[u] = distances[v] + path_maxes[v, u]
            optimal_paths[u] = optimal_paths[v][:]
            optimal_paths[u].append(u)

optimal_path = optimal_paths[-1]








