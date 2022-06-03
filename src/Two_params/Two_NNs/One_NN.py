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
import scipy.optimize as op

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# standard symbolic algebra module
#import sympy as sm
#sm.init_printing()

# module to save results
import joblib as jb
# pytorch
import torch
import torch.nn as nn
import matplotlib as mp

#from torch.utils.data import Dataset

# split data into a training set and a test set
from sklearn.model_selection import train_test_split

# to reload modules
# import importlib

# written out by LFI_generate1.ipynb
import LFIutil as lfi

# update fonts
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set usetex = False if LaTex is not 
# available on your system or if the 
# rendering is too slow
mp.rc('text', usetex=True)

# set a seed to ensure reproducibility
seed = 128
rnd  = np.random.RandomState(seed)

datafile =  'DATA_FOR_TWO_NNs.csv'
data = pd.read_csv(datafile)
#select a subset of rows
data = data.iloc[:5000,:]

MLE  = True
#WHICH is a name for the run, so that e.g. plots could be saved using this name
if MLE:
    target = 'Z_MLE_TRUE'
    WHICH  = 'MLE'
else:
    target = 'Z_MLE_FALSE   '
    WHICH  = 'nonMLE'
    
source = ['theta', 'nu', 'N', 'M']

# print(np.allclose(np.array(data.Z_MLE_TRUE), np.array(data.Z_MLE_FALSE) ) )

XMIN  = 0
XMAX  = 20
XBINS = 200
NU    = 3
D     = [(1, 0), (2, 0), (3, 0), 
         (1, 1), (2, 1), (3, 1)]

def hist_data(nu, N, M,
              xbins=XBINS,
              xmin=XMIN, 
              xmax=XMAX,
              mle=MLE,
              Ndata=100000):

    theta = st.uniform.rvs(xmin, xmax, size=Ndata)
    n = st.poisson.rvs(theta + nu)
    m = st.poisson.rvs(nu, size=Ndata)
    Z = (lfi.t2(theta, n, m, mle) < 
         lfi.t2(theta, N, M, mle)).astype(np.int32)

    xrange = (xmin, xmax)

    # weighted histogram   (count the number of ones per bin)
    y1, bb = np.histogram(theta, 
                          bins=xbins, 
                          range=xrange, 
                          weights=Z)
    
    # unweighted histogram (count number of ones and zeros per bin)
    yt, _ = np.histogram(theta, 
                         bins=xbins, 
                         range=xrange)

    y =  y1 / yt    
    
    return y, bb




# Fraction of the data assigned as test data
fraction = 1/102
# Split data into a part for training and a part for testing
train_data, test_data = train_test_split(data, 
                                         test_size=fraction)

# Split the training data into a part for training (fitting) and
# a part for validating the training.
fraction = 1/101
train_data, valid_data = train_test_split(train_data, 
                                          test_size=fraction)

# reset the indices in the dataframes and drop the old ones
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
test_data  = test_data.reset_index(drop=True)

train_t, train_x = lfi.split_t_x(train_data, target, source)
valid_t, valid_x = lfi.split_t_x(valid_data, target, source)
test_t,  test_x  = lfi.split_t_x(test_data,  target, source)

model = lfi.Model()
learning_rate = 1.e-3
optimizer     = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate) 

#############################TRACES
traces = ([], [], [])
traces_step = 10

n_batch       = 50
n_iterations  = 2000

traces = lfi.train(model, optimizer, lfi.average_loss,
               lfi.get_batch,
               train_x, train_t, 
               valid_x, valid_t,
               n_batch, 
               n_iterations,
               traces,
               step=traces_step)

n_batch       = 500
n_iterations  = 1000

traces = lfi.train(model, optimizer, lfi.average_loss,
               lfi.get_batch,
               train_x, train_t, 
               valid_x, valid_t,
               n_batch, 
               n_iterations,
               traces,
               step=traces_step)

##############################

def usemodel(nu, N, M,
             xbins=XBINS,
             xmin=XMIN,
             xmax=XMAX):
    
    xstep = (xmax-xmin) / xbins
    bb    = np.arange(xmin, xmax+xstep, xstep)
    X     = (bb[1:] + bb[:-1])/2
    X     = torch.Tensor([[x, nu, N, M] for x in X])
    
    model.eval()
    return model(X).detach().numpy(), bb

if __name__ == '__main__':

    #plot_data(NU, D) 

    # print('train set size:        %6d' % train_data.shape[0])
    # print('validation set size:   %6d' % valid_data.shape[0])
    # print('test set size:         %6d' % test_data.shape[0])
    # print(train_data[:5][source])
    # print(train_t.shape, train_x.shape)
    # print(model)
    # lfi.plot_average_loss(traces)
    lfi.plot_data(NU, D, 
          func=usemodel, 
          gfile='fig_model_vs_DL_%s.png' % WHICH) 
