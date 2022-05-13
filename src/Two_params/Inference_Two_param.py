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
# %matplotlib inline
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
    """Use the svg format to display a plot in Jupyter for sharper images!"""
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


k=1
def L_prof(n,m,theta):
    k=1
    k1 = k+1
    k2 = 0.5/k1
    g = n+m - k1*theta
    nu_hat = k2* (g+ np.sqrt(g*g +4*k1*m*theta))
    p1 = st.poisson.pmf(n, mu = theta + nu_hat)
    p2 = st.poisson.pmf(m, mu = k * nu_hat)
    
    return p1*p2


model = torch.load( 'TWO_PARAMETERS_TRAINED _MODEL.pth')

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


#FIX N AND M and theta and nu
Bprime=1000000
N, M = 3, 2
model=model

def sim_nm_lambdas(theta, nu):
    """Sample n ~ Pois(theta+nu), m ~ Pois(nu), and compute lambda_gen(theta, n, m) and lambda_D(theta, N, M)
    return (n, m, lambda_gen, lambda_D)"""
    n = st.poisson.rvs(theta+nu, size=Bprime)
    m = st.poisson.rvs(nu, size=Bprime)
    lambda_gen = utils.lambda_test(theta, n, m, MLE)
    lambda_D = utils.lambda_test(theta, N, M, MLE)
    return (n, m, lambda_gen, lambda_D)
    
# points =(theta, nu)
def sim_nm_lambdas_diff_theta_nu(points):
    """for each theta and nu tuple, generate n,m, lambda_gen, lambda_D
    then calculate P(\lambda_{gen}<\lambda_D)$ Exact and \hat{p}(\theta,\nu,N,M) \approx P(\lambda_{gen}<\lambda_D)"""
    results_list = []
    for p in points:
        theta, nu = p
        n, m, lambda_gen, lambda_D = sim_nm_lambdas(theta, nu)
        p_lambda_gen_lt_lambda_D = (lambda_gen <= lambda_D).astype(np.int32)
        
        X = np.array([theta, nu, N, M]) 
        #RESHAPE X (FOR INFERENCE) TO THE SAME SHAPE AS THE DATA IT TRAINED ON (like train_data), which has shape(nsamples, nfeatures)
        X=X.reshape(1,-1)

        X =torch.Tensor(X)
        # print(X.shape)
        
        with torch.no_grad():
            model.eval()

            # X = torch.Tensor(X)
            phat = model(X)
            phat = phat.detach().flatten()
            # phat=phat.numpy()
            PHAT = phat.view(-1).numpy()
            # PHAT=1-PHAT
        #results will be an array of n, an array of m, an array of lambda_gen, and array of lambda_D
        # an array of P_Exact, an array pf phat
        results_list.append((n, m, lambda_gen, lambda_D, theta, nu, p_lambda_gen_lt_lambda_D, PHAT))
    return results_list


def plot_one(p_lambda_gen_lt_lambda_D, PHAT, theta, nu, ax):
    ftsize = 20
    xmin, xmax= 0, 1.2
    ymin, ymax = 0, 50
    x_range = (xmin, xmax)
    y_range = (ymin, ymax)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    NBINS=40
    ax.set_xlabel(r'$P(\lambda_{gen}<\lambda_D)$',fontsize=ftsize)
    ax.hist(p_lambda_gen_lt_lambda_D, bins=NBINS, range=x_range, density=True,
            alpha=0.4, histtype='stepfilled', edgecolor='black', label=r'$P(\lambda_{gen}<\lambda_D)$ Exact with $\hat{\theta}_{MLE}$')
    ax.hist(PHAT, bins=NBINS, range=x_range, density=True,
            alpha=0.4, histtype='stepfilled', edgecolor='black',label=r'$\hat{p}(\theta,\nu,N,M) \approx P(\lambda_{gen}<\lambda_D)$ ')
    
    
    #positions for text
    xwid = (xmax-xmin)/12
    ywid = (ymax-ymin)/12
    xpos = xmin + xwid*2
    ypos = ymin + ywid*6
    ax.text(xpos, ypos,
    r'$ \theta = %d, \nu = %d$' % (theta, nu),
    fontsize=ftsize)
    ax.legend()

def plot_all(results, fgsize=(15,15)):
    plt.figure(figsize=fgsize)
    fig, ax = plt.subplots(2, 3, figsize=fgsize)
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(wspace=0.2)
    ax = ax.flatten()
    #result is (n, m, lambda_gen, lambda_D, theta, nu p_lambda_gen_lt_lambda_D, PHAT)
    for i, result in enumerate(results):
        n, m, lambda_gen, lambda_D, theta, nu, p_lambda_gen_lt_lambda_D, PHAT =result
        plot_one(p_lambda_gen_lt_lambda_D, PHAT, theta, nu, ax[i])
        
    for j in range(len(results), len(ax)):
        ax[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    MLE=True
    #generate (theta, nu) points
    points = [(theta, nu) for theta, nu in 
            zip(np.random.randint(low=1,high=20,size=6), np.random.randint(low=1,high=20,size=6))]
    print('points =', points, '\n')
    results = sim_nm_lambdas_diff_theta_nu(points)
    plot_all(results)