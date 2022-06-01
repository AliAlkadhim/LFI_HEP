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

# standard symbolic algebra module
#import sympy as sm
#sm.init_printing()

# module to save results
import joblib as jb
# pytorch
import torch
import torch.nn as nn
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

MLE  = True
if MLE:
    target = 'Z_MLE_TRUE'
    WHICH  = 'MLE'
else:
    target = 'Z_MLE_FALSE   '
    WHICH  = 'nonMLE'
    
source = ['theta', 'nu', 'N', 'M']

# print(np.allclose(np.array(data.Z_MLE_TRUE), np.array(data.Z_MLE_FALSE) ) )
