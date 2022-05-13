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
# update fonts
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set a seed to ensure reproducibility
# seed = 128
# rnd  = np.random.RandomState(seed)
#matplotlib configurations
def use_svg_display(): 
    """Use the svg format to display a plot in Jupyter for sharper imagesget_ipython().getoutput(""""")
    backend_inline.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5, 2.5)):  
    """Set the figure size for matplotlib."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
    
def DR(s, theta):
    return sp.special.gammainc(s, theta)

def DL(s, theta):
    return 1 - sp.special.gammainc(s+1, theta)
import random


model = torch.load( 'TWO_PARAMETERS_TRAINED _MODEL.pth')

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


N, M = 2, 3
points = [(theta, nu) for theta, nu in 
        zip(np.random.randint(low=0,high=4,size=6), np.random.randint(low=0,high=4,size=6))]
theta, nu = points[0]

X = np.array([theta, nu, N, M]) 
X=X.reshape(1,-1)

X =torch.Tensor(X)
X.shape



torch.normal(0, 1, (3, 4))


def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)# noise term
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
features[:5], labels[:5]


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


x = torch.arange(4.0)
x


x.requires_grad_(True)
x


x.grad?


x.grad#default is none as expected


y = 2 * torch.dot(x, x)
y


y.backward()
x.grad


x


x.grad == 4 * x


# PyTorch accumulates the gradient in default, we need to clear the previous
# values
x.grad.zero_()
x


#calculate the gradient of another function


y = x.sum()
y.backward()
x.grad


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
plt.plot(x.detach(), y.detach(), label=r'$ReLU(x) = max(x,0)$')
plt.xlabel('x')
y.backward(torch.ones_like(x), retain_graph=True)
plt.plot(x.detach(), x.grad, label=r'$\frac{\partial \ ReLU(x)}{\partial x}$')
plt.legend()


y = torch.sigmoid(x)
plt.plot(x.detach(), y.detach(), label=r'Sigmoid $(x)$')
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plt.plot(x.detach(), x.grad, label=r'$\frac{\partial \ Sigmoid(x)}{\partial x}$')
plt.xlabel(r'$x$'); plt.legend()


y = torch.tanh(x)
plt.plot(x.detach(), y.detach(), label=r'tanh $(x)$')
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plt.plot(x.detach(), x.grad, label=r'$\frac{\partial \ tanh(x)}{\partial x}$')
plt.xlabel(r'$x$'); plt.legend()



