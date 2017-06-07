#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:14:37 2017

@author: angus
"""

# Given a matrix of weights between edges, goal is to find the shortes path from the first point to the end
# No edge between vertices is represented by 999999
# Paths are from row number to column number

import numpy as np

# Dijkstra
# set parameters
L = 15 # number of vertices

# set weights
w = np.random.uniform(1, 10, 15*15).reshape(15, 15)
w = w - np.diag(np.diag(w))

# set initial conditions
unvisited = list(range(L))

distances = [999999]*L
distances[0] = 0

paths = [[0]]*L

while len(unvisited) > 0:
    i = np.argmin([distances[i] for i in unvisited])
    v = unvisited[i]
    unvisited.pop(i)
    for u in range(L):
        if distances[u] > distances[v] + w[v, u]:
            distances[u] = distances[v] + w[v, u]
            paths[u] = paths[v][:]
            paths[u].append(u)




