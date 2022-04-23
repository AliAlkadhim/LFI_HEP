import numpy as np
import scipy as sp
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as mp
import matplotlib.pyplot as plt
# force inline plots
get_ipython().run_line_magic("matplotlib", " inline")
plt.style.use('seaborn-deep')
import torch.nn as nn
import copy
import pandas as pd

font = {'family' : 'fantasy',
        'weight' : 'normal',
        'size'   : 18
        }
mp.rc('font', **font)
mp.rc('text', usetex=True)
mp.rc('lines', linewidth=4)


inference_df = pd.read_csv('data/results/inference_df.csv')
inference_df.describe()


plt.scatter(inference_df.theta, inference_df.calulated_p, label='calculated p-value')
# plt.scatter(inference_df.theta, inference_df.phat, label=r'$\hat{p}$')
plt.legend()


np.sum([st.poisson.pmf(k=N, mu=theta) for N in range(1)])


1-sp.special.gammainc(1+1, theta)


fig, ax = plt.subplots(1,2,figsize=(15,7))
data_df = pd.read_csv('data/UNIFORM_5K_CDFD_eq_1.csv')
hist_counts_weighted, b_weighted = np.histogram(data_df.theta, weights=np.array(data_df.Z), bins=100)
hist_counts_unweighted, b_unweighted = np.histogram(data_df.theta, bins=100)

hist_counts= hist_counts_weighted/hist_counts_unweighted
b= b_weighted#/b_unweighted

bin_centers = (b[1:]+b[:-1])/2
# import boost_histogram as bh
# hist = bh.Histogram(bh.axis.Regular(bins=100, start=0, stop=10))
calculated_p_value_poisson_sum, calculated_p_value_gamma =[],[]
D=1
for theta in bin_centers:
    calculated_p_value_gamma.append((1-sp.special.gammainc(D+1, theta)))
    p_cal=np.sum([st.poisson.pmf(k=N, mu=theta) for N in range(D)])
    calculated_p_value_poisson_sum.append(p_cal)
    
ax[0].plot(bin_centers, hist_counts, label='data hist')

ax[0].plot(bin_centers, calculated_p_value_poisson_sum, label='calculated p-value with Poisson sum')
ax[0].plot(bin_centers, calculated_p_value_gamma, label='calculated p-value with gamma')

ax[0].legend()

ax[1].scatter(data_df.theta, (1-sp.special.gammainc(1+1, data_df.theta)))
ax[1].set_xlabel(r'data $\theta$'); ax[1].set_ylabel('calculated p-value')


import boost_histogram as bh
bins, edges = bh.Histogram(bh.axis.Regular(100,0,10))

bins



