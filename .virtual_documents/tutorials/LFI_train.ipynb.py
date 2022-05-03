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
import sympy as sm
sm.init_printing()

#  a function to save results
import joblib as jb

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset

#  split data into a training set and a test set
from sklearn.model_selection import train_test_split

# to reload modules
import importlib

get_ipython().run_line_magic("matplotlib", " inline")


# update fonts
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set a seed to ensure reproducibility
seed = 128
rnd  = np.random.RandomState(seed)


datafile = 'data.db'
print('loading get_ipython().run_line_magic("s'", " % datafile)")
data = jb.load(datafile)

inputs = ['theta', 'D']

print('N_{Z=1}/N_{Z, all} = ', sum(data.Z)/len(data.Z))
print(data[:5])


def DR(s, theta):
    return sp.special.gammainc(s, theta)

def DL(s, theta):
    return 1 - sp.special.gammainc(s+1, theta)


# Check that histogrammed data agrees with exact calculation of DL.

xmin, xmax = 0, 20
xrange= (xmin, xmax)
xbins = 40
xstep = (xmax - xmin) / xbins
x     = np.arange(xmin+0.5*xstep, xmax + 0.5*xstep, xstep)

def hist_data(data, D, x):
    
    select = data.D == D

    # weighted histogram   (count the number of ones per bin)
    y1, _ = np.histogram(data.theta[select], 
                         bins=xbins, 
                         range=xrange, 
                         weights=data.Z[select]) 

    # unweighted histogram (count number of ones and zeros per bin)
    yt, _ = np.histogram(data.theta[select], 
                         bins=xbins, 
                         range=xrange)

    # approximation of DL(D, x)
    y =  y1 / yt    

    # exact
    p = DL(D, x)
    
    return (y, p)


def plot_data(data, func, Dmin, Dmax, x, 
              gfile='fig_data.png', 
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
    
    for j, d in enumerate(range(Dmin, Dmax+1)):
        
        # compute DL 
        y, p = func(data, d, x)
        # "func" is the function used as an input, in this case it is hist_data, which returns (y, p), where y is weighted_histogram/unweighted_histoggram, p is the exact p value
        
        ax[j].set_xlim(xmin, xmax)
        ax[j].set_ylim(0, 1)
        ax[j].set_xlabel(r'$\theta$', fontsize=FONTSIZE)
        ax[j].set_ylabel(r'$E(Z|\theta)$', fontsize=FONTSIZE)
        
        ax[j].plot(x, y, 'b', lw=2, label='approx')
        ax[j].plot(x, p, 'r', lw=2, label='exact')
        
        ax[j].grid(True, which="both", linestyle='-')
        ax[j].text(12, 0.42, r'$D = get_ipython().run_line_magic("d$'", " % d, fontsize=FONTSIZE) ")

        ax[j].legend(loc='upper right')
        
    # hide unused sub-plots
    for k in range(Dmax+1-Dmin, len(ax)):
        ax[k].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(gfile)
    plt.show()


Dmin, Dmax = 0, 5
plot_data(data, hist_data, Dmin, Dmax, x) 


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

print('train set size:        get_ipython().run_line_magic("6d'", " % train_data.shape[0])")
print('validation set size:   get_ipython().run_line_magic("6d'", " % valid_data.shape[0])")
print('test set size:         get_ipython().run_line_magic("6d'", " % test_data.shape[0])")

train_data[:5][inputs]


def split_t_x(df, inp=inputs):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    t = np.array(df['Z'])
    x = np.array(df[inp])
    return (t, x)

train_t, train_x = split_t_x(train_data)
valid_t, valid_x = split_t_x(valid_data)
test_t,  test_x  = split_t_x(test_data)


def get_batch(x, t, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = rnd.choice(len(x), batch_size)
    batch_x = x[rows]
    batch_t = t[rows]
    return (batch_x, batch_t)


# Note: there are several average loss functions available 
# in pytorch, but it's useful to know how to create your own.
def average_loss(f, t):
    # f and t must be of the same shape
    return  torch.mean((f - t)**2)


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
    print("get_ipython().run_line_magic("10s\t%10s\t%10s"", " % \")
          ('iteration', 'train-set', 'valid-set'))
    
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
                print("get_ipython().run_line_magic("10d\t%10.6f\t%10.6f"", " % \")
                      (xx[-1], acc_t, acc_v))
            else:
                xx.append(xx[-1] + step)
                print("\rget_ipython().run_line_magic("10d\t%10.6f\t%10.6f"", " % \")
                      (xx[-1], acc_t, acc_v), end='')
                
            yy_t.append(acc_t)
            yy_v.append(acc_v)
    print()      
    return (xx, yy_t, yy_v)


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


get_ipython().run_cell_magic("writefile", " dnnmodel.py", """
import torch
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self, n_inputs=2, n_nodes=20, n_layers=5):

        # call constructor of base (or super, or parent) class
        super(Model, self).__init__()

        self.layers = []
        
        # create input layer
        self.layer0 = nn.Linear(n_inputs, n_nodes)
        self.layers.append(self.layer0)

        # create "hidden" layers
        for l in range(1, n_layers):
            cmd = 'self.layer%d = nn.Linear(%d, %d)' % \
            (l, n_nodes, n_nodes)
            exec(cmd)
            cmd = 'self.layers.append(self.layer%d)' % l
            exec(cmd)
          
        # create output layer
        cmd = 'self.layer%d = nn.Linear(%d, 1)' % (n_layers, n_nodes)
        exec(cmd)
        cmd = 'self.layers.append(self.layer%d)' % n_layers
        exec(cmd)

    # define (required) method to compute output of network
    def forward(self, x):
        y = x
        for layer in self.layers[:-1]:
            y = layer(y)
            y = torch.relu(y)
        y = self.layers[-1](y)
        y = torch.sigmoid(y)
        return y
""")


import dnnmodel
importlib.reload(dnnmodel)
model = dnnmodel.Model()
print(model)


learning_rate = 1.e-3
optimizer     = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate) 

traces = ([], [], [])
traces_step = 10


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


def usemodel(data, d, x):
    model.eval()
    xx= [[z, d] for z in x]
    X = torch.Tensor(xx)
    y = model(X).detach().numpy()
    p = DL(d, x)
    return (y, p)


plot_data(data, usemodel, Dmin, Dmax, x, gfile='fig_model_vs_DL.png') 



