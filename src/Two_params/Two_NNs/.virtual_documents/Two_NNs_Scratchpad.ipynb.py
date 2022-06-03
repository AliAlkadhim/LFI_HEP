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
import utils
import joblib as jb

# update fonts
FONTSIZE = 18
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size":FONTSIZE,
})

# set a seed to ensure reproducibility
# seed = 128
# rnd  = np.random.RandomState(seed)
    
import random


f, axes = plt.subplots(1, 3, sharey=True, figsize=(13,7))
axes = axes.flatten()
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
########RELU
relu = torch.relu(x)
axes[0].plot(x.detach(), relu.detach(),label=r'$ReLU(x) \equiv max(x,0)$')
#take derivative
relu.backward(torch.ones_like(x), retain_graph=True)
axes[0].plot(x.detach(), x.grad, label=r'$\frac{\partial \ ReLU(x)}{\partial x}$')
######SIGMOID
sigmoid = torch.sigmoid(x)
axes[1].plot(x.detach(), sigmoid.detach(),label=r'Sigmoid $(x)$')
# Clear out previous gradients
x.grad.data.zero_()
#take derivative
sigmoid.backward(torch.ones_like(x), retain_graph=True)
axes[1].plot(x.detach(), x.grad, label=r'$\frac{\partial \ Sigmoid(x)}{\partial x}$')
###tanh
tanh = torch.tanh(x)
axes[2].plot(x.detach(), tanh.detach(), label=r'tanh $(x)$')
x.grad.data.zero_()
tanh.backward(torch.ones_like(x),retain_graph=True)
axes[2].plot(x.detach(), x.grad, label=r'$\frac{\partial \ tanh(x)}{\partial x}$')

for ax in axes:
    ax.set_xlabel('$x$')
    ax.set_ylim(-0.2,2.5)
    ax.legend(fontsize=15)


MLE_TRUE_FILE = '../../tutorials/data1.db'
MLE_FALSE_FILE = '../../tutorials/data2.db'
#For both, Z for the two parameter problem is Z2
MLE=True

if MLE:
    datafile = MLE_TRUE_FILE
else:
    datafile = MLE_FALSE_FILE
    
data = jb.load(MLE_FALSE_FILE)
features = data[['theta', 'nu', 'N', 'M']]
target = data['Z2']
features



