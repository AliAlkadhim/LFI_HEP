#!/usr/bin/env python
# coding: utf-8

# # Likelihood-Free Inference - Model Training
# 
# Ali Al Kadhim and Harrison B. Prosper<br>
# Department of Physics, Florida State University<br>
# Date: 12 May 2022
# 
# ## Introduction
# 
# See __LFI_generate_data.ipynb__ and __LFI_train.ipynb__. 
# 
# ## The Poisson Problem: A Single Count
# 
# For 1-parameter problems is is always possible to compute confidence intervals with exact coverage (see LFI_train.ipynb). This is true, in particular, for the statistical model
# \begin{align}
# \textrm{P}(n | \theta) & = \frac{e^{-\theta} \theta^n}{n!}
# \end{align}
# 
# In notebook __LFI_train.ipynb__, we used the statistic $t(n) = n$. Here, we use the statistic
# 
# $
# \begin{align}
#   t & = -2 \ln \left[ \frac{L_p(\theta)}{L_p(\hat{\theta})} \right]
#   = -2 \ln \left[ \frac{e^{-\theta} \theta^n}{e^{-n} n^n} \right],\\
#   & = 2(\theta - n) + 2 n \ln (n \, / \, \theta ),
# \end{align}
# $
# 
# which is the prototype of the most commonly used statistic for multi-parameter problems (see LFI_train2.ipynb). This statistic is used because *asymptotically*, that is, when the data are numerous or counts are large, the sampling distribution of $t$ and its variants can be computed analytically [2]. 
# 
# 
# ### References
#   1. Anne Lee et al., https://arxiv.org/abs/2107.03920
#   2. Glen Cowan, Kyle Cranmer, Eilam Gross and Ofer Vitells, *Asymptotic formulae for likelihood-based tests of new physics*,  	Eur.Phys.J.C71:1554,2011.
# 

# In[1]:


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
import importlib

# written out by LFI_generate1.ipynb
import LFIutil as lfi

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


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


# ### Load data

# In[3]:


datafile = 'data1.db'
print('loading %s' % datafile)
data = jb.load(datafile)

target = 'Z2'
source = ['theta', 'N']

data[:5]


# ### Plot data

# In[4]:


# Check that histogrammed data agrees with exact calculation of DL.

XMIN  = 0
XMAX  = 20
XBINS = 200
D     = [0, 1, 2, 3, 4, 5]

def hist_data(N,
              xbins=XBINS,
              xmin=XMIN, 
              xmax=XMAX,
              Ndata=100000):

    theta = st.uniform.rvs(xmin, xmax, size=Ndata)
    n = st.poisson.rvs(theta)
    Z = (lfi.t1(theta, n) < 
         lfi.t1(theta, N)).astype(np.int32)

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


# In[20]:


def plot_data(D, 
              func=None,
              xbins=XBINS,
              xmin=XMIN, 
              xmax=XMAX,
              gfile='fig_data1.png', 
              fgsize=(10, 6)):
    
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
    
    for j, N in enumerate(D):
        
        y, bb = hist_data(N)
    
        ax[j].set_xlim(xmin, xmax)
        ax[j].set_ylim(0, 1)
        ax[j].set_xlabel(r'$\theta$', fontsize=FONTSIZE)
        ax[j].set_ylabel(r'$E(Z|\theta)$', fontsize=FONTSIZE)
        
        x = (bb[1:]+bb[:-1])/2
        ax[j].plot(x, y, 'b', lw=2, label='approx')
        
        if func:
            p, _ = func(N)
            ax[j].plot(x, p, 'r', lw=2, label='model')
        
        ax[j].grid(True, which="both", linestyle='-')
        ax[j].text(10.1, 0.42, r'$N = %d$' % N, 
                   fontsize=FONTSIZE) 

        ax[j].legend(loc='upper right')
        
    # hide unused sub-plots
    for k in range(j+1, len(ax)):
        ax[k].set_visible(False)
    
    plt.tight_layout()
    print('saving..', gfile)
    plt.savefig(gfile)
    plt.show()


# In[6]:


plot_data(D) 


# ### Train, validation, and test sets
# There is some confusion in terminology regarding validation and test samples (or sets). We shall adhere to the defintions given here https://machinelearningmastery.com/difference-test-validation-datasets/):
#    
#   * __Training Dataset__: The sample of data used to fit the model.
#   * __Validation Dataset__: The sample of data used to decide 1) whether the fit is reasonable (e.g., the model has not been overfitted), 2) decide which of several models is the best and 3) tune model hyperparameters.
#   * __Test Dataset__: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.
# 
# The validation set will be some small fraction of the training set and will be used to decide when to stop the training.

# In[7]:


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

print('train set size:        %6d' % train_data.shape[0])
print('validation set size:   %6d' % valid_data.shape[0])
print('test set size:         %6d' % test_data.shape[0])

train_data[:5][source]


# Split data into targets $t$ and inputs $\mathbf{x}$

# In[8]:


def split_t_x(df, target, source):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    t = np.array(df[target])
    x = np.array(df[source])
    return t, x

train_t, train_x = split_t_x(train_data, target, source)
valid_t, valid_x = split_t_x(valid_data, target, source)
test_t,  test_x  = split_t_x(test_data,  target, source)


# ### Return a (random) batch of data from the training set

# In[9]:


def get_batch(x, t, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = rnd.choice(len(x), batch_size)
    batch_x = x[rows]
    batch_t = t[rows]
    return (batch_x, batch_t)


# ### Empirical risk (that is, average loss)
# 
# The empirical risk, which is the __objective function__ we shall minimize, is defined as
# 
# \begin{align}
# R_M(\theta) & = \frac{1}{M}\sum_{m=1}^M L(t_m, f_m),
# \end{align}
# 
# where 
# 
# \begin{align*}
#     f_m & \equiv f(\mathbf{x}_m, \theta),\\ \\ \textrm{and} \\
#     L(t, f) &= (t - f)^2
# \end{align*}
# 
# The empirical risk $R_M$ approximates the __risk__
# 
# \begin{align}
# R[f] & = \int \cdots \int \, p(t, \mathbf{x}) \, L(t, f(\mathbf{x}, \theta)) \, dt \, d\mathbf{x},
# \end{align}
# 
# which is a __functional__ of the model $f$. The quantity $p(t, \mathbf{x}) \, dt\, d\mathbf{x}$ is the probability distribution from which the sample $\{ (t_m, \mathbf{x}_m), m = 1,\cdots, M \}$ is presumed to have been drawn. 

# In[10]:


# Note: there are several average loss functions available 
# in pytorch, but it's useful to know how to create your own.
def average_loss(f, t):
    # f and t must be of the same shape
    return  torch.mean((f - t)**2)


# This function is used to validate the model while the it is being fitted.

# In[11]:


def validate(model, avloss, inputs, targets):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode
    
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        x = torch.from_numpy(inputs).float()
        t = torch.from_numpy(targets).float()
        # remember to reshape!
        o = model(x).reshape(t.shape)
    return avloss(o, t)


# ### Function to execute training loop

# In[12]:


def train(model, optimizer, avloss, getbatch,
          train_x, train_t, 
          valid_x, valid_t,
          batch_size, 
          n_iterations, traces, 
          step=50):
    
    # to keep track of average losses
    xx, yy_t, yy_v = traces
    
    n = len(valid_x)
    
    print('Iteration vs average loss')
    print("%10s\t%10s\t%10s" %           ('iteration', 'train-set', 'valid-set'))
    
    for ii in range(n_iterations):

        # set mode to training so that training specific 
        # operations such as dropout are enabled.
        model.train()
        
        # get a random sample (a batch) of data (as numpy arrays)
        batch_x, batch_t = getbatch(train_x, train_t, batch_size)
        
        # convert the numpy arrays batch_x and batch_t to tensor 
        # types. The PyTorch tensor type is the magic that permits 
        # automatic differentiation with respect to parameters. 
        # However, since we do not need to take the derivatives
        # with respect to x and t, we disable this feature
        with torch.no_grad(): # no need to compute gradients 
            # wrt. x and t
            x = torch.from_numpy(batch_x).float()
            t = torch.from_numpy(batch_t).float()      

        # compute the output of the model for the batch of data x
        # Note: outputs is 
        #   of shape (-1, 1), but the tensor targets, t, is
        #   of shape (-1,)
        # In order for the tensor operations with outputs and t
        # to work correctly, it is necessary that they have the
        # same shape. We can do this with the reshape method.
        outputs = model(x).reshape(t.shape)
   
        # compute a noisy approximation to the average loss
        empirical_risk = avloss(outputs, t)
        
        # use automatic differentiation to compute a 
        # noisy approximation of the local gradient
        optimizer.zero_grad()       # clear previous gradients
        empirical_risk.backward()   # compute gradients
        
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer.step()            # move one step
        
        if ii % step == 0:
            
            acc_t = validate(model, avloss, train_x[:n], train_t[:n]) 
            acc_v = validate(model, avloss, valid_x[:n], valid_t[:n])

            if len(xx) < 1:
                xx.append(0)
                print("%10d\t%10.6f\t%10.6f" %                       (xx[-1], acc_t, acc_v))
            else:
                xx.append(xx[-1] + step)
                print("\r%10d\t%10.6f\t%10.6f" %                       (xx[-1], acc_t, acc_v), end='')
                
            yy_t.append(acc_t)
            yy_v.append(acc_v)
    print()      
    return (xx, yy_t, yy_v)


# In[13]:


def plot_average_loss(traces):
    
    xx, yy_t, yy_v = traces
    
    # create an empty figure
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)

    ax.set_title("Average loss")
    
    ax.plot(xx, yy_t, 'b', lw=2, label='Training')
    ax.plot(xx, yy_v, 'r', lw=2, label='Validation')

    ax.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax.set_ylabel('average loss', fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')
    ax.legend(loc='upper right')

    plt.show()


# ### Define model $f(\mathbf{x}, \theta)$

# In[14]:


get_ipython().run_cell_magic('writefile', 'dnnmodel1.py', '\nimport torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    \n    def __init__(self, n_inputs=2, n_nodes=20, n_layers=5):\n\n        # call constructor of base (or super, or parent) class\n        super(Model, self).__init__()\n\n        self.layers = []\n        \n        # create input layer\n        self.layer0 = nn.Linear(n_inputs, n_nodes)\n        self.layers.append(self.layer0)\n\n        # create "hidden" layers\n        for l in range(1, n_layers):\n            cmd = \'self.layer%d = nn.Linear(%d, %d)\' % \\\n            (l, n_nodes, n_nodes)\n            exec(cmd)\n            cmd = \'self.layers.append(self.layer%d)\' % l\n            exec(cmd)\n          \n        # create output layer\n        cmd = \'self.layer%d = nn.Linear(%d, 1)\' % (n_layers, n_nodes)\n        exec(cmd)\n        cmd = \'self.layers.append(self.layer%d)\' % n_layers\n        exec(cmd)\n\n    # define (required) method to compute output of network\n    def forward(self, x):\n        y = x\n        for layer in self.layers[:-1]:\n            y = layer(y)\n            y = torch.relu(y)\n        y = self.layers[-1](y)\n        y = torch.sigmoid(y)\n        return y')


# In[15]:


import dnnmodel1
importlib.reload(dnnmodel1)
model = dnnmodel1.Model()
print(model)


# ### Train!

# Instantiate an optimizer, then train

# In[16]:


learning_rate = 1.e-3
optimizer     = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate) 

traces = ([], [], [])
traces_step = 10


# In[17]:


n_batch       = 50
n_iterations  = 20000

traces = train(model, optimizer, average_loss,
               get_batch,
               train_x, train_t, 
               valid_x, valid_t,
               n_batch, 
               n_iterations,
               traces,
               step=traces_step)

n_batch       = 500
n_iterations  = 10000

traces = train(model, optimizer, average_loss,
               get_batch,
               train_x, train_t, 
               valid_x, valid_t,
               n_batch, 
               n_iterations,
               traces,
               step=traces_step)

plot_average_loss(traces)


# In[18]:


def usemodel(N,
             xbins=XBINS,
             xmin=XMIN,
             xmax=XMAX):
    
    xstep = (xmax-xmin) / xbins
    bb    = np.arange(xmin, xmax+xstep, xstep)
    X     = (bb[1:] + bb[:-1])/2
    X     = torch.Tensor([[x, N] for x in X])
    
    model.eval()
    return model(X).detach().numpy(), bb


# In[21]:


plot_data(D, 
          func=usemodel, 
          gfile='fig_model_vs_DL_1.png') 


# In[ ]:




