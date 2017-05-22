#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:03:40 2017

@author: angus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Wan, Kveton and Ashkan 2015 CombLinTS algorithm
# Should work better than other algorithms when L is very large as doesn't depend on L
# Makes use of Lxd generalization matrix Φ of features for each element of E
# w̄ lies on or close to subspace span of Φ
# define θ∗ = argmin_θ ||w̄ − Φθ||_2
# could think of features as for example knowledge of other activities/events which affects outcome??


# Scenario: User knows Φ matrix of activities/events
# At each stage user samples coefficients θ used to estimate w̄
# and uses this estimate with oracle to determine which move to play
# observes weight for each play and updates parameters of θ sampling accordingly

### Set parameters
L = 40
K = 8
iters = 100
lambda_true = 1 # (used to generate true coefficients of Φ which are then used to calculate w̄ )
sigma_true = 0.5 # (used when adding noise to true w at each iteration)
d = 10
phi = np.random.normal(0, 1, (L,d))
theta_opt = np.random.multivariate_normal([0]*d, np.diag([lambda_true**2]*d))

# create matrices to store results at each iteration
theta = np.zeros((d, iters+1))
w_est = np.zeros((L, iters))
A = np.zeros((L, iters))
w_actual = np.zeros((L, iters))

# model parameters
lambda_est = 1
sigma_est = 0.5

### INIT
t=0
cov_mat = [np.diag([lambda_est**2]*d)]*(iters+1)
theta[:,t] = [0]*d

### iterate over number of iterations
for t in range(iters):
    # draw sample theta and use to estimate w and then choose K best
    theta_sample = np.random.multivariate_normal(theta[:,t], cov_mat[t])
    w_est[:,t] = np.dot(phi, theta_sample)
    A[np.argsort(w_est[:,t])[-K:,],t] = 1
    
    # calcualate actual w based on theta_opt plus some noise
    w_actual_noise = np.random.normal(0, sigma_true**2, L)
    w_actual[:,t] = np.dot(phi, theta_opt) + w_actual_noise
    
    # update covariance matrix and theta
    # init
    theta[:,(t+1)] = theta[:,t]
    cov_mat[t+1] = cov_mat[t]
    
    # iterating over played elements (should this do it in order best to worst or something?)
    for k in np.argsort(w_est[:,t])[-K:,]:
        # update theta
        theta[:,(t+1)] = (
                np.dot(np.diag([1]*d) - 
                       np.dot(np.dot(cov_mat[t+1], np.transpose(phi[k,]).reshape(d, 1)), phi[k,].reshape(1, d)) /
                       (np.dot(np.dot(phi[k,], cov_mat[t+1]), phi[k,]) + sigma_est**2)
                       ,
                       theta[:,(t+1)].reshape(d,1))
                +
                (np.dot(cov_mat[t+1], np.transpose(phi[k,]).reshape(d, 1)) /
                 (np.dot(np.dot(phi[k,], cov_mat[t+1]), phi[k,]) + sigma_est**2)
                 * w_actual[k, t])
                ).reshape(d)
        
        # update covariance matrix
        cov_mat[t+1] = (cov_mat[t+1] -
               np.dot(np.dot(np.dot(cov_mat[t+1], np.transpose(phi[k,]).reshape(d, 1)), phi[k,].reshape(1, d)), cov_mat[t+1]) /
               (np.dot(np.dot(phi[k,], cov_mat[t+1]), phi[k,]) + sigma_est**2))

# plotting path of plays
plt.pcolor(A)

# getting optimum strategy to play each round by calculating highest expected w using theta_opt
A_opt = np.zeros(L)
A_opt[np.argsort(np.dot(phi, theta_opt))[-K:]] = 1

# calculating regret
regret = np.sum(A_opt.reshape(40,1)*w_actual) - np.sum(A*w_actual)


### Bundling into function to test parameters
def combLinTS(K, L, iterations, lambda_true, sigma_true, d, lambda_est, sigma_est):
    ### Set parameters
    phi = np.random.normal(0, 1, (L,d))
    theta_opt = np.random.multivariate_normal([0]*d, np.diag([lambda_true**2]*d))
    
    # create matrices to store results at each iteration
    theta = np.zeros((d, iterations+1))
    w_est = np.zeros((L, iterations))
    A = np.zeros((L, iterations))
    w_actual = np.zeros((L, iterations))
    
    ### INIT
    t=0
    cov_mat = [np.diag([lambda_est**2]*d)]*(iterations+1)
    theta[:,t] = [0]*d
    
    ### iterate over number of iterations
    for t in range(iterations):
        # draw sample theta and use to estimate w and then choose K best
        theta_sample = np.random.multivariate_normal(theta[:,t], cov_mat[t])
        w_est[:,t] = np.dot(phi, theta_sample)
        A[np.argsort(w_est[:,t])[-K:,],t] = 1
        
        # calcualate actual w based on theta_opt plus some noise
        w_actual_noise = np.random.normal(0, sigma_true**2, L)
        w_actual[:,t] = np.dot(phi, theta_opt) + w_actual_noise
        
        # update covariance matrix and theta
        # init
        theta[:,(t+1)] = theta[:,t]
        cov_mat[t+1] = cov_mat[t]
        
        # iterating over played elements (should this do it in order best to worst or something?)
        for k in np.argsort(w_est[:,t])[-K:,]:
            # update theta
            theta[:,(t+1)] = (
                    np.dot(np.diag([1]*d) - 
                           np.dot(np.dot(cov_mat[t+1], np.transpose(phi[k,]).reshape(d, 1)), phi[k,].reshape(1, d)) /
                           (np.dot(np.dot(phi[k,], cov_mat[t+1]), phi[k,]) + sigma_est**2)
                           ,
                           theta[:,(t+1)].reshape(d,1))
                    +
                    (np.dot(cov_mat[t+1], np.transpose(phi[k,]).reshape(d, 1)) /
                     (np.dot(np.dot(phi[k,], cov_mat[t+1]), phi[k,]) + sigma_est**2)
                     * w_actual[k, t])
                    ).reshape(d)
            
            # update covariance matrix
            cov_mat[t+1] = (cov_mat[t+1] -
                   np.dot(np.dot(np.dot(cov_mat[t+1], np.transpose(phi[k,]).reshape(d, 1)), phi[k,].reshape(1, d)), cov_mat[t+1]) /
                   (np.dot(np.dot(phi[k,], cov_mat[t+1]), phi[k,]) + sigma_est**2))
    
    # getting optimum strategy to play each round by calculating highest expected w using theta_opt
    A_opt = np.zeros(L)
    A_opt[np.argsort(np.dot(phi, theta_opt))[-K:]] = 1
    
    # calculating regret
    regret = np.sum(A_opt.reshape(L,1)*w_actual) - np.sum(A*w_actual)
    
    return(regret)


### Plot regret as a function of L (should be linear)
ls = list(range(100, 1001, 100))
regrets_l = [0]*len(ls)
for l in range(len(ls)):
    for _ in range(100):
        regrets_l[l] += combLinTS(K=10, L=ls[l], iterations=100, lambda_true=1,
                 sigma_true=0.5, d=10, lambda_est=1, sigma_est=0.5)/100

plt.scatter(x=ls, y=regrets_l)
plt.title('regret as function of cardinality of ground set (L)')
plt.xlabel('L')
plt.ylabel('regret')
plt.show()
plt.close()


### Plot regret as a function of K (should be linear)
ks = list(range(1, 51, 5))
regrets_k = [0]*len(ks)
for k in range(len(ks)):
    for _ in range(100):
        regrets_k[k] += combLinTS(K=ks[k], L=100, iterations=100, lambda_true=1,
                 sigma_true=0.5, d=10, lambda_est=1, sigma_est=0.5)/100

plt.scatter(x=ks, y=regrets_k)
plt.title('regret as function of number of chosen items (K)')
plt.xlabel('K')
plt.ylabel('regret')
plt.show()
plt.close()


########################################
### Do same for combLinUCB algorithm ###
########################################
# similar setup but selection of A at each iteration is based on UCB instead of sampling
# needs extra parameter c which controls the 'degree of optimism'
# high c means higher optimism and so more exploration
def combLinUCB(K, L, iterations, lambda_true, sigma_true, d, lambda_est, sigma_est, c):
    ### Set parameters
    phi = np.random.normal(0, 1, (L,d))
    theta_opt = np.random.multivariate_normal([0]*d, np.diag([lambda_true**2]*d))
    
    # create matrices to store results at each iteration
    theta = np.zeros((d, iterations+1))
    w_est = np.zeros((L, iterations))
    A = np.zeros((L, iterations))
    w_actual = np.zeros((L, iterations))
    
    ### INIT
    t=0
    cov_mat = [np.diag([lambda_est**2]*d)]*(iterations+1)
    theta[:,t] = [0]*d
    
    ### iterate over number of iterations
    for t in range(iterations):
        # draw sample theta and use to estimate w and then choose K best
        w_est[:,t] = np.dot(phi, theta[:,t]) + c*np.sqrt(np.diag(np.dot(np.dot(phi, cov_mat[t]), np.transpose(phi)))) # THIS IS THE ONLY LINE CHANGED FROM TS ALGORITHM
        A[np.argsort(w_est[:,t])[-K:,],t] = 1
        
        # calcualate actual w based on theta_opt plus some noise
        w_actual_noise = np.random.normal(0, sigma_true**2, L)
        w_actual[:,t] = np.dot(phi, theta_opt) + w_actual_noise
        
        # update covariance matrix and theta
        # init
        theta[:,(t+1)] = theta[:,t]
        cov_mat[t+1] = cov_mat[t]
        
        # iterating over played elements (should this do it in order best to worst or something?)
        for k in np.argsort(w_est[:,t])[-K:,]:
            # update theta
            theta[:,(t+1)] = (
                    np.dot(np.diag([1]*d) - 
                           np.dot(np.dot(cov_mat[t+1], np.transpose(phi[k,]).reshape(d, 1)), phi[k,].reshape(1, d)) /
                           (np.dot(np.dot(phi[k,], cov_mat[t+1]), phi[k,]) + sigma_est**2)
                           ,
                           theta[:,(t+1)].reshape(d,1))
                    +
                    (np.dot(cov_mat[t+1], np.transpose(phi[k,]).reshape(d, 1)) /
                     (np.dot(np.dot(phi[k,], cov_mat[t+1]), phi[k,]) + sigma_est**2)
                     * w_actual[k, t])
                    ).reshape(d)
            
            # update covariance matrix
            cov_mat[t+1] = (cov_mat[t+1] -
                   np.dot(np.dot(np.dot(cov_mat[t+1], np.transpose(phi[k,]).reshape(d, 1)), phi[k,].reshape(1, d)), cov_mat[t+1]) /
                   (np.dot(np.dot(phi[k,], cov_mat[t+1]), phi[k,]) + sigma_est**2))
    
    # getting optimum strategy to play each round by calculating highest expected w using theta_opt
    A_opt = np.zeros(L)
    A_opt[np.argsort(np.dot(phi, theta_opt))[-K:]] = 1
    
    # calculating regret
    regret = np.sum(A_opt.reshape(L,1)*w_actual) - np.sum(A*w_actual)
    
    return(regret)


### Plot regret as a function of L (should be linear)
ls = list(range(100, 1001, 100))
regrets_l_UCB = [0]*len(ls)
for l in range(len(ls)):
    for _ in range(100):
        regrets_l_UCB[l] += combLinUCB(K=10, L=ls[l], iterations=100, lambda_true=1,
                     sigma_true==0.5, d=10, lambda_est=1, sigma_est=0.5, c=1)/100

plt.scatter(x=ls, y=regrets_l_UCB)
plt.title('regret as function of cardinality of ground set (L)')
plt.xlabel('L')
plt.ylabel('regret')
plt.show()
plt.close()


### Plot regret as a function of K (should be linear)
ks = list(range(1, 51, 5))
regrets_k_UCB = [0]*len(ks)
for k in range(len(ks)):
    for _ in range(100):
        regrets_k_UCB[k] += combLinUCB(K=ks[k], L=100, iterations=100, lambda_true=1,
                     sigma_true=0.5, d=10, lambda_est=1, sigma_est=0.5, c=1)/100

plt.scatter(x=ks, y=regrets_k_UCB)
plt.title('regret as function of number of chosen items (K)')
plt.xlabel('K')
plt.ylabel('regret')
plt.show()
plt.close()










































