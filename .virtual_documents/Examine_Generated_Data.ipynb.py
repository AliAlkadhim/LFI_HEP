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


inference_df = pd.read_csv('data/results/inference_df.csv')
inference_df


plt.scatter(inference_df.theta, inference_df.calulated_p, label='calculated p-value')
# plt.scatter(inference_df.theta, inference_df.phat, label=r'$\hat{p}$')
plt.legend()


np.sum([st.poisson.pmf(k=N, mu=theta) for N in range(1)])


1-sp.special.gammainc(1+1, theta)


fig, ax = plt.subplots(1,2,figsize=(20,7))
data_df = pd.read_csv('data/UNIFORM_5K_CDFD_eq_3.csv')
hist_counts, b = np.histogram(data_df.theta, density=True, weights=data_df.Z, bins=100)
bin_centers = (b[1:]+b[:-1])/2
calculated_p_value =[]
D=3
for theta in bin_centers:
    # calculated_p_value.append((1-sp.special.gammainc(1+1, theta)))
    p_cal=np.sum([st.poisson.pmf(k=N, mu=theta) for N in range(D)])
    calculated_p_value.append(p_cal)
ax[0].plot(bin_centers, hist_counts, label='data hist')

ax[0].plot(bin_centers, calculated_p_value, label='calculated')
ax[0].legend()
ax[1].scatter(data_df.theta, (1-sp.special.gammainc(1+1, data_df.theta)))
ax[1].set_xlabel(r'data $\theta$'); ax[1].set_ylabel('calculated p-value')



