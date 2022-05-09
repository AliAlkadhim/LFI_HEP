import os, sys

# the standard module for tabular data
import pandas as pd

# the standard module for array manipulation
import numpy as np

# the standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt

# standard scientific python module
import scipy as sp
import scipy.stats as st



#  a function to save results
import joblib as jb

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset

#  split data into a training set and a test set
from sklearn.model_selection import train_test_split

# update fonts
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set a seed to ensure reproducibility
seed = 128
rnd  = np.random.RandomState(seed)
#########################################################


def DL(s, theta):
    return 1 - sp.special.gammainc(s+1, theta)

def hist_data_one_param(data, D, x):
    """Given a dataframe data which has columns theta and D, return a histogram of theta at those D values with the
    weights Z, divided by a histogram for theta at those D values without the weights. This will be the the approximate p value at bins x. The exact p value for D at x is calculated with DL"""
    mask = data.D == D

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
    approx_p =  y1 / yt    

    # exact
    exact_p = DL(D, x)
    
    return (approx_p, exact_p)

def plot_data_one_param(data, func, Dmin, Dmax, x, 
              gfile='fig_data.png', 
              fgsize=(10, 6)):
    """func is something that returns (approx_p, exact_p), as in hist_data_one_param above"""
    # make room for 6 sub-plots
    fig, ax = plt.subplots(nrows=2, 
                           ncols=3, 
                           figsize=fgsize)
    
    # padding
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(wspace=0.20)
    
    # use flatten() to convert a numpy array of 
    # shape (nrows, ncols) to a 1-d array. 
    ax = ax.flatten()
    
    for j, d in enumerate(range(Dmin, Dmax+1)):
        
        # compute DL 
        approx_p, exact_p = func(data, d, x)#d is value of D in this range
        # "func" is the function used as an input, in this case it is hist_data, which returns (y, p), where y is weighted_histogram/unweighted_histoggram, p is the exact p value
        
        ax[j].set_xlim(xmin, xmax)
        ax[j].set_ylim(0, 1)
        ax[j].set_xlabel(r'$\theta$', fontsize=FONTSIZE)
        ax[j].set_ylabel(r'$E(Z|\theta)$', fontsize=FONTSIZE)
        
        ax[j].plot(x, approx_p, 'b', lw=2, label='approx')
        ax[j].plot(x, exact_p, 'r', lw=2, label='exact')
        
        ax[j].grid(True, which="both", linestyle='-')
        ax[j].text(12, 0.42, r'$D = %d$' % d, fontsize=FONTSIZE) 

        ax[j].legend(loc='upper right')
        
    # hide unused sub-plots
    for k in range(Dmax+1-Dmin, len(ax)):
        ax[k].set_visible(False)
    
    plt.tight_layout()
    # plt.savefig(gfile)
    plt.show()


# plot_data_one_param(data, hist_data_one_param, Dmin, Dmax, x) 

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
    model_at_inputs =data.phat[mask]

    print('\napprox_p by histogramming =',approx_p_by_hist)
    print('\napprox_p by model =',model_at_inputs)
    return approx_p_by_hist
    # return (approx_p, exact_p)



if __name__ == '__main__':
    # datafile = '../data/data.db'
    # datafile = pd.read_csv('../data/two_parameters_N_M_Uniformly_sampled_1M.csv')
    datafile = pd.read_csv('../data/results/INFERENCE_DF_TWOPARAMS_1M.csv')
    # print('loading %s' % datafile)
    # data = jb.load(datafile)
    # datafile['D'] = np.ones(len(datafile))
    data=datafile

    # inputs = ['theta', 'D']

    # print(data[:5])
    Mmin, Mmax = 0, 10
    Nmin, Nmax = 0, 10
    xmin, xmax = 0, 20
    xrange= (xmin, xmax)
    xbins = 40
    xstep = (xmax - xmin) / xbins
    x     = np.arange(xmin+0.5*xstep, xmax + 0.5*xstep, xstep)
    N, M = 12, 2
    hist_data_two_params(data, N, M, x)
    
