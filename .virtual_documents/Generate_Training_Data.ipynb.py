import numpy as np
import scipy as sp
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as mp
import matplotlib.pyplot as plt
# force inline plots
get_ipython().run_line_magic("matplotlib", " inline")
plt.style.use('seaborn-deep')
import torch.nn as nn
import copy
import pandas as pd





Bprime=1000000
D = 9 #this does not work for large values of D
L_obs=30 
#b= mean background
print('The size of B: ', Bprime)
print('The observed signal signal N (or bold X in the paper): ', D)
print('The observed luminosity: ', L_obs)
# print('The observed background'


def L(X, theta):
    """likelihood with one parameter"""
    return st.poisson.pmf(X, mu=theta)
    
def labd_one_param(X, theta):
    num = L(X, theta)
    den = L(X, D)
    return -2 * np.log(num/den)


#T=[[theta_i],[Z_i]]
def generate_training_data(Bprime, D):
    
    T = [[],[]]
    for i in range(Bprime):
        theta = st.expon.rvs() #sample theta from an exponential distribution
        #theta has to be positive because its an input to a poisson. This prior should also be close to the cound D

        N = np.random.poisson(lam=theta) #draw count samples randomly from a poisson distribution
        #this X is really N
        lam_observed = labd_one_param(X=D, theta=theta)#
        lam_i = labd_one_param(X=X, theta=theta)
        if lam_i < lam_true:
            Z_i=1
        else:
            Z_i=0
        T[0].append(theta)
        T[1].append(Z_i)
        
        return np.array(T[0]), np.array(T[1])


def generate_training_data_one_parameter(Bprime, D, save_data=False):
    #T=[[theta_i],[Z_i]]
    T = [[],[]]
    for i in range(Bprime):
        theta = st.expon.rvs() #sample theta from an exponential distribution
        #theta has to be positive because its an input to a poisson. This prior should also be close to the cound D

        N = np.random.poisson(lam=theta) #draw count samples randomly from a poisson distribution
        #this X is really N

        if D < N:
            Z_i=1
        else:
            Z_i=0
        T[0].append(theta)
        T[1].append(Z_i)
        
    if save_data:
        Training_data_1_param = {'theta' : T[0], 'Z' : T[1]}
        Training_data_1_param = pd.DataFrame.from_dict(Training_data_1_param)
        Training_data_1_param.to_csv('Training_data_1_param.csv')
        
    return np.array(T[0]), np.array(T[1])


theta, Z = generate_training_data_one_parameter(Bprime=100000, D=9)
np.sum(Z)
