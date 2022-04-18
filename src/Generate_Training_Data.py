import numpy as np
import scipy as sp
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as mp
import matplotlib.pyplot as plt
# force inline plots
# %matplotlib inline
plt.style.use('seaborn-deep')
import torch.nn as nn
import copy
import pandas as pd
import sys
import argparse

parser=argparse.ArgumentParser(description='generate training data')
parser.add_argument('--p', type=int, help='the number of parameters')
parser.add_argument('--D', type=int, help='the value of D')
parser.add_argument('--Bprime', type=int, help='the value of Bprime, ie the size of the dataset')

args = parser.parse_args()

number_of_params = args.p
D = args.D

#Bprime=1000000




def L(X, theta):
    """likelihood with one parameter"""
    if number_of_params==1:
        return st.poisson.pmf(X, mu=theta)

def lambd(X, theta):
    """test statistic lambda for one parameter likelihood"""
    if number_of_params==1:
        num = L(X, theta)
        den = L(X, D)
        return -2 * np.log(num/den)

#T=[[theta_i],[Z_i]]
def generate_training_data(Bprime, D, save_data=False):
    
    T = [[],[]]
    for i in range(Bprime):
        #the st.expon.rvs returns just one sample (number) from the exponential distribution, which is what we want
        theta = st.expon.rvs() #sample theta from an exponential distribution,
        #theta has to be positive because its an input to a poisson. This prior should also be close to the cound D
        
        #Now use the sampler F_theta to generate X (X ~ F_theta)
        N = np.random.poisson(lam=theta)  #this is the F_theta sampler
        #draw count samples randomly from a poisson distribution
        #this X is really N
        #CALCULATE LAMBDA WITH X FIXED (as D): this is called lambda_obs
        lambda_observed = lambd(X=D, theta=theta)#
        #CALCULATE LAMBDA WITH X BEING SAMPLED
        lambda_i = lambd(X=N, theta=theta)
        if lambda_i < lambda_observed:
            Z_i=1
        else:
            Z_i=0
        T[0].append(theta)
        T[1].append(Z_i)
        
        
        Training_data_1_param = {'theta' : T[0], 'Z' : T[1]}
            
        Training_data_1_param = pd.DataFrame.from_dict(Training_data_1_param)
        if save_data:
            Training_data_1_param.to_csv('data/Training_data_1_param_using_lambda_1M.csv')

    
    print(np.array(T[0]), np.array(T[1]))
    return np.array(T[0]), np.array(T[1])


if __name__ == '__main__':
    generate_training_data(Bprime=args.Bprime, D=args.D)