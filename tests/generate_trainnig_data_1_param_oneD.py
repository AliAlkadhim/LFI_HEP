import numpy as np
import scipy as sp
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as mp
import matplotlib.pyplot as plt
# force inline plots
plt.style.use('seaborn-deep')
import torch.nn as nn
import copy
import pandas as pd


Bprime=1000
D = 1 #this does not work for large values of D





def generate_training_data_one_parameter(Bprime, D, save_data=False):
    #T=[[theta_i],[Z_i]]
    T = [[],[]]
    for i in range(Bprime):
        #theta has to be positive because its an input to a poisson. This prior should also be close to the cound D
        theta = st.expon.rvs()
        N = st.poisson.rvs(theta) #draw count samples randomly from a poisson distribution
        #this X is really N

        if D <= N:
            Z_i=1
        else:
            Z_i=0
        T[0].append(theta)
        T[1].append(Z_i)
        
    if save_data:
        Training_data_1_param = {'theta' : T[0], 'Z' : T[1]}
        Training_data_1_param = pd.DataFrame.from_dict(Training_data_1_param)
        # Training_data_1_param.to_csv('data/Training_data_1_param_1M_D_eq_1.csv')
    print(Training_data_1_param)
    return Training_data_1_param
# theta, Z = generate_training_data_one_parameter(Bprime=Bprime, D=D, save_data=False)

def phat(theta):
    D=1
    count=0

    for i in range(Bprime):
        N = np.random.poisson(theta)
        if N >=D:
            count +=1
    return count/Bprime

# p_calc = sp.special.gammainc(D, theta)
Training_data_1_param = generate_training_data_one_parameter(Bprime, D, save_data=True)
# theta = Training_data_1_param.theta
phat_l = []; p_calculated=[]
for ind, row in Training_data_1_param.iterrows():

    #print(row.theta)
    #create new column phat
    
    phat_i = phat(row.theta)
    phat_l.append(phat_i)
    p_calculated.append(sp.special.gammainc(D, row.theta))

Training_data_1_param['phat'] = phat_l
Training_data_1_param['p_calculated'] = p_calculated

print(Training_data_1_param.head())



# if __name__ == '__main__':
#     generate_training_data_one_parameter(Bprime, D, save_data=True)