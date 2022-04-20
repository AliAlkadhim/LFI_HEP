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
import os

TRAINING_DATA = os.environ.get("TRAINING_DATA")
RUN_NAME = os.environ.get("RUN_NAME")

parser=argparse.ArgumentParser(description='generate training data')
parser.add_argument('--p', type=int, help='the number of parameters', required=False)
parser.add_argument('--D', type=int, help='the value of D')
# parser.add_argument('--Bprime', type=int, help='the value of Bprime, ie the size of the dataset')

args = parser.parse_args()

number_of_params = args.p
D = int(args.D)

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
def generate_training_data(Bprime, D, save_data=True):
    
    T = [[],[]]
    for i in range(Bprime):
        #the st.expon.rvs returns just one sample (number) from the exponential distribution, which is what we want
        theta = np.random.uniform(low=0.5, high=5) #sample theta from an exponential distribution,
        #theta has to be positive because its an input to a poisson. This prior should also be close to the cound D
        
        #Now use the sampler F_theta to generate X (X ~ F_theta)
        N = np.random.poisson(lam=theta)  #this is the F_theta sampler
        
        if D <= N:
            Z_i=1
        else:
            Z_i=0
        T[0].append(theta)
        T[1].append(Z_i)
        #draw count samples randomly from a poisson distribution
        #this X is really N
        #CALCULATE LAMBDA WITH X FIXED (as D): this is called lambda_obs
        # lambda_observed = lambd(X=D, theta=theta)#
        # #CALCULATE LAMBDA WITH X BEING SAMPLED
        # lambda_i = lambd(X=N, theta=theta)
        # if lambda_i < lambda_observed:
        #     Z_i=1
        # else:
        #     Z_i=0
        # T[0].append(theta)
        # T[1].append(Z_i)
        
        
        data_1_param = {'theta' : T[0], 'Z' : T[1]}
            
        data_1_param = pd.DataFrame.from_dict(data_1_param)
        if save_data:
            data_1_param.to_csv('data/'+RUN_NAME+'D_eq_%d.csv' % D)
            # two_thirds = 2*len(data_1_param)//3
            # TRAIN_DATA_1_PARAM = data_1_param.iloc[:two_thirds,]
            # TEST_DATA_1_PARAM = data_1_param.iloc[two_thirds:,]

            # TRAIN_DATA_1_PARAM.to_csv('data/TRAIN_DATA_1_PARAM_D_eq_%d.csv' %D)
            # TEST_DATA_1_PARAM.to_csv('data/TEST_DATA_1_PARAM_D_eq_%d.csv' %D)

    
    print(np.array(T[0]), np.array(T[1]))
    return np.array(T[0]), np.array(T[1])


if __name__ == '__main__':
    generate_training_data(Bprime=100000, D=args.D)
    #now go to the main directory and do something like python src/Generate_Training_Data_One_param.py --D 1
    print('Data generation is done')