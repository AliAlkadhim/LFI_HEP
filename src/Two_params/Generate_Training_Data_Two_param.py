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

def theta_hat(n,m, MLE=True):
    theta_hat = n-m
    
    if not MLE:
        theta_hat = theta_hat * (theta_hat > 0)
    return theta_hat

def lambda_test(theta, n, m, MLE=True):
    Ln = L_prof(n,m,theta)
    Ld = L_prof(n,m, theta_hat(n,m, MLE))
    lambda_  = -2*np.log(Ln/Ld)
    return lambda_

chi2_exp_size=4000

def run_sim( theta, nu):
    """Sample n ~ Pois(theta+nu), m ~ Pois(nu), and compute lambda(theta, n, m)"""
    n = st.poisson.rvs(theta+nu, size=chi2_exp_size)
    m = st.poisson.rvs(nu, size=chi2_exp_size)
    lambda_ = lambda_test(theta, n, m, MLE)
    return (n, m, lambda_)

def run_sims(points):
    """Run an entire simulation (that is, generate n and m from run_sim above, and calculate lambda) for each point, where a point is a tuple of (theta, nu)"""
    lambda_results=[]

    for p in points:
        theta, nu = p
        n, m, lambda_ = run_sim(theta, nu)
        lambda_results.append((n, m, lambda_, theta, nu))
        print((theta, nu))
    return lambda_results


def plot_one(lambda_, theta, nu, ax):
    """Histogram the CDF of  lambda_t = -2log(Lp(theta)/Lp(theta_hat)), for a given (fixed) theta and nu.
    Also, plot the actual CDF of a chi^2 distribution with 1 free parameter (since only theta is left after we profile nu) """
    ftsize = 16
    xmin= 0
    xmax= 10
    ymin= 0
    ymax= 1
    x_range = (xmin, xmax)
    y_range = (ymin, ymax)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(r'$\lambda (\theta, n, m, \hat{\nu}(\theta))$',fontsize=ftsize)
    ax.set_ylabel(r'cdf$(\lambda)$', fontsize=ftsize)
    ax.hist(lambda_, bins=5*xmax, range=x_range,
    color=(0.8,0.8,0.9),
    density=True, cumulative=True,
    histtype='stepfilled', edgecolor='black')
    
    x = np.arange(0, xmax, 0.2)
    y = st.chi2.cdf(x, 1)
    ax.plot(x, y, color='blue',
    linewidth=2)
    # annotate
    xwid = (xmax-xmin)/12
    ywid = (ymax-ymin)/12
    xpos = xmin + xwid/2
    ypos = ymin + ywid*2
    ax.text(xpos, ypos,
    r'$ \theta = %d, \nu = %d$' % (theta, nu),
    fontsize=ftsize)
    
def plot_all(results, fgsize=(10,6)):
    plt.figure(figsize=fgsize)
    fig, ax = plt.subplots(2, 3, figsize=fgsize)
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(wspace=0.2)
    ax = ax.flatten()
    
    for i, result in enumerate(results):
        n, m, lambda_, theta, nu = result
        plot_one(lambda_, theta, nu, ax[i])
    for j in range(len(results), len(ax)):
        ax[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()
    
def generate_training_data(Bprime, save_data=False):
    """Generate the training data, that is, features=[theta, nu, N, M], targets=Z"""
    #sample theta and nu from uniform(0,20)
    theta = st.uniform.rvs(thetaMin, thetaMax, size=Bprime)
    nu = st.uniform.rvs(nuMin, nuMax, size=Bprime)
    #n,m ~ F_{\theta,\nu}, ie our simulator. sample n from a Poisson with mean theta+nu 
    n = st.poisson.rvs(theta+ nu, size=Bprime)
    #sample m from a poisson with mean nu
    m = st.poisson.rvs(nu, size=Bprime)
    
    #sample our observed counts (N,M), which take the place of D
    N = np.random.randint(Nmin, Nmax, size=Bprime)
    M = np.random.randint(Mmin, Mmax, size=Bprime)
    print('n=', n)
    print('m=', m)
    print('N=', N)
    print('M=', M)
    lambda_gen = lambda_test(theta, n, m, MLE)
    print('lambda_gen= ', lambda_gen)
    lambda_D = lambda_test(theta, N, M, MLE)
    print('lambda_D= ', lambda_D)
    #if lambda_gen <= lambda_D: Z=1, else Z=0
    Z = (lambda_gen <= lambda_D).astype(np.int32)
    
    data_2_param = {'Z' : Z, 'theta' : theta, 'nu': nu, 'N':N, 'M':M}

    data_2_param = pd.DataFrame.from_dict(data_2_param)
    if save_data:
        data_2_param.to_csv('TWO_PARAMETERS_TRAINING_DATA_1M.csv')

    print(data_2_param.head())
    return data_2_param


if __name__ == '__main__':
    MLE=True
    #generate (theta, nu) points
    points = [(theta, nu) for theta, nu in 
            zip(np.random.randint(low=1,high=4,size=6), np.random.randint(low=0,high=4,size=6))]

    results = run_sims(points)
    print(results[1])
    plot_all(results)

Bprime    = 200000
thetaMin, thetaMax =  0, 20
nuMin, nuMax = 0, 20
Mmin, Mmax =  0 , 10
Nmin, Nmax =  0,10
MLE=True
data_2_param=generate_training_data(Bprime,save_data=True)