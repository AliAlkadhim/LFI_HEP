import numpy as np
import scipy as sp
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as mp
mp.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
# force inline plots
# get_ipython().run_line_magic("matplotlib", " inline")
plt.style.use('seaborn-deep')
import torch.nn as nn
import copy
import pandas as pd
import sys
import os
# update fonts
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set a seed to ensure reproducibility
# seed = 128
# rnd  = np.random.RandomState(seed)
def DR(s, theta):
    return sp.special.gammainc(s, theta)

def DL(s, theta):
    return 1 - sp.special.gammainc(s+1, theta)
from IPython.display import Image, display
algorithm2 = Image('src/images/Algorithm2.jpg')
display(algorithm2)


get_ipython().run_line_magic("run", " GenerateData_and_Train_2_parameters_step_1.ipynb")


datafile = pd.read_csv('data/results/INFERENCE_DF_TWOPARAMS_1M.csv')
data=datafile[['Z', 'theta', 'nu', 'N', 'M', 'phat']]
data.head()


model = torch.load('models/Regressor_TwoParams_theta_nu_m_n.pth'); model.parameters


def hist_data_two_params(data, N, M, x):
    """Given a dataframe data which has columns theta and M and N, return a histogram of theta at those [M,N] values with the
    weights Z, divided by a histogram for theta at those D values without the weights. This will be the the approximate p value at bins x. The exact p value for D at x is calculated with DL"""
    # mask = (data['value2'] == 'some_string') & (data['value'] > 4)
    mask = (data.M == M) & (data.N == N)

    # weighted histogram   (count the number of ones per bin)
    y1, _ = np.histogram(data.theta[mask], 
                         bins=xbins, 
                         range=xrange, 
                         weights=data.Z[mask]) 

    # unweighted histogram (count number of ones and zeros per bin)
    yt, _ = np.histogram(data.theta[mask], 
                         bins=xbins, 
                         range=xrange)

    # approximation of DL(D, x)
    approx_p_by_hist =  y1 / yt    

    # exact
    # exact_p = DL(D, x)
    # inputs = ['theta', 'nu', 'N', 'M']
    # df_at_inputs = data[inputs]
    # df_at_inputs = df_at_inputs[mask]
    data_at_mask = data[mask]
    inputs = ['theta', 'nu', 'N', 'M']
    # data_df='data/two_parameters_N_M_Uniformly_sampled_1M.csv'
    # df = pd.read_csv(data_df)
    Input_features = data_at_mask[inputs]
    ############ ML inference part
    with torch.no_grad():
        model.eval()

        X = torch.Tensor(Input_features.values)
        phat = model(X)
        phat = phat.detach().flatten()
        # phat=phat.numpy()
        PHAT = phat.view(-1).numpy()
        print(phat.flatten().shape)

    # model_at_inputs =data.phat[mask]
    model_at_inputs = PHAT
    
    print('\napprox_p by histogramming =',approx_p_by_hist)
    print('\napprox_p by model =',model_at_inputs)
    return approx_p_by_hist, model_at_inputs 


xmin, xmax = 0, 20
xrange= (xmin, xmax)
xbins = 40
xstep = (xmax - xmin) / xbins
x     = np.arange(xmin+0.5*xstep, xmax + 0.5*xstep, xstep)
N, M = 12, 2
approx_p_by_hist, model_at_inputs = hist_data_two_params(data, N, M, x)


approx_p_by_hist.shape, model_at_inputs.shape


plt.plot(x, approx_p_by_hist, label='approx_p_by_hist')
plt.plot(x, model_at_inputs, label='approx_p_by_hist')
plt.legend()



