#!/usr/bin/env python
# coding: utf-8

# # Likelihood-Free Inference - Data Generation
# 
# Ali Al Kadhim and Harrison B. Prosper<br>
# Department of Physics, Florida State University<br>
# Date: 22 April 2022<br>
# Updated: 12 May 2022
# 
# 
# ## Introduction
# 
# The jargon likelihood-free inference (LFI) is a tad unfortunate. For any Monte Carlo-based simulation there is clearly an underlying statistical (or probability) model from which we are implicitly sampling. 
# The __likelihood function__, that is, the statistical model after data have been entered into it exists in the same sense that $\pi$ exists. While it is impossible to write down all of the digits of $\pi$, these digits are presumed to exist nevertheless!  The likelihood, though typically intractable, exists in the same sense.
# 
# A better name, perhaps, for this inference approach is __Simulation-Based Inference (SBI)__, which better expresses the idea that one makes inferences directly from accurate simulations of the data generation mechanism without the need for explicit knowledge of the likelihood. The key idea is that the ensemble of simulations constitute a point-cloud approximation to the probability model. 
# 
# LFI focuses directly on the statistical quantities that are of direct interest such as p-values and confidence intervals in a frequentist setting or summaries of posterior densities such as the first few moments or its quantiles. The key insight (see, for example, Ann Lee *et al.* [1]) is that it is generally easier to compute integrals of densities than the densities themselves.
# 
# In this series of notebooks, we apply an algorithm from Ref.[1] to a classic problem in statistics. See notebooks __LFI_train.ipynb__ and __LFI_train1.ipynb__, which for the 1-parameter Poisson problem uses two different __statistics__. A statistic is any function of
# of potential observations. See notebook LFI_train.ipynb for more details.
# 
# 
# In this notebook we generate two datasets __data1.db__ and __data2.db__ that are used as training data in notebooks __LFI_train.ipynb__, __LFI_train1.ipynb__ and __LFI_train2.ipynb__.
# 
# 
# ### Datasets for the 1-parameter model
# 
# For an experiment that yields a single count the standard statistical model is a Poisson distribution,
# 
# $
# \begin{align}
# \textrm{P}(n | \theta) & = \textrm{Poisson}(n , \theta) \equiv \frac{e^{-\theta} \theta^n}{n!}
# \end{align}
# $
# 
# When data (here a single datum $n = N$) are entered into a statistical model the latter becomes, by definition, a __likelihood function__.
# 
# The dataset, __data1.db__, associated with this model comprises
# a large set of 3-tuples $(Z_i, \theta_i, N_i)$ where 
# 
# $
# \begin{align}
# \theta_i & \sim \textrm{uniform}(0, 20), \\
# n_i & \sim \textrm{poisson}(\theta_i),\\
# N_i & \sim \textrm{randint}(0, 10), \textrm{ and } \\
# \textrm{ and } \\
# Z_i & = I[ t(n_i) \lt t(N_i) ],  
# \end{align}
# $
# 
# $I$ is the indicator function and $t$ is one of two statistics:
# 
# $
# \begin{align}
#     t & = n_i, \\
#     \text{or } t & = -2 \ln \left[ \frac{L_p(\theta)}{L_p(\hat{\theta})} \right],
# \end{align}
# $
# 
# 
# where 
# 
# $L_p(\theta)$ is the __profile likelihood__. 
# The parameters of a statistical model and, therefore, of a likelihood function, are typically divided into __parameters of interest__ (__poi__) and the rest. The latter are often referred to as  __nuisance parameters__. When the nuisance parameters in the likelihood function are replaced by estimates, typically, by their __maximum likelihood estimates__ (MLE) for *given* values of the parameters of interest, the resulting function is referred to as a __profile likelihood__. Think of the likelihood as a mountain range; then the profile likelihood is the outline of the mountain range's projection against the sky. 
# 
# The profile likelihood is an important quantity because it can be used to compute intervals with approximate coverage provided that certain conditions hold, the main one being that the __estimator__, $\hat{\theta}$, (that is, a function or procedure that yields estimates) yields estimates that do not lie on the boundary of the parameter space. In the 2-parameter statistical model, described next, the maximum likelihood estimator yields intervals with good, albeit approximate, coverage. However, achieving good coverage with an estimator that explicitly violates that condition proves to be challenging.
# 
# 
# ### Datasets for the 2-parameter model
# 
# The following 2-parameter, 2-count, model 
# 
# $
# \begin{align}
# \textrm{P}(n, m | \theta, \nu) & = \frac{e^{-(\theta + \nu)} (\theta + \nu)^n}{n!}
# \frac{e^{-\nu} \nu^m}{m!},
# \end{align}
# $
# 
# is the prototype of many statistical models in astronomy and particle physics in which data are binned and the count in each bin consists *a priori* of the sum of counts from a signal source and background source with unknown mean counts $\theta$ and $\nu$, respectively. In astronomy, the model is called the ON/OFF model, where ON refers to telescope time on a patch of sky in which a signal source may be present and which yields a signal plus background count $n = N$, while in the simplest case OFF refers to the *same* amount of telescope time on a patch of sky similar to the ON patch except that a signal source is nor present. The OFF patch yields count $m = M$. The goal is to make inferences about the parameter $\theta$.
# 
# The two datasets associated with this model consist of a large set of 5-tuples $(Z_i, \theta_i, \nu_i, N_i, M_i)$ where 
# 
# $
# \begin{align}
# \theta_i & \sim \textrm{uniform}(0, 20), \\
# \nu_i & \sim \textrm{uniform}(0, 20), \\
# n_i & \sim \textrm{poisson}(\theta_i + \nu_i),\\
# m_i & \sim \textrm{poisson}(\nu_i),\\
# N_i & \sim \textrm{randint}(0, 10), \\
# M_i & \sim \textrm{randint}(0, 10), \textrm{ and } \\
# \textrm{ and } \\
# Z & = I[ t(n_i, m_i) \lt t(N_i, M_i) ].
# \end{align}
# $
# 
# As in the single-count case, we consider two statistics $t$. The first uses the MLE of $\theta$ namely $\hat{\theta} = n - m$ and yields $Z_i = Z_{1i}$ while the second $Z_i = Z_{2i}$ is computed
#  using the non-MLE of $\theta$ in which $\hat{\theta} = n - m$ if $n > m$ and is equal to $0$ otherwise.
# 
# Datasets __data1.db__ is used in __LFI_train.ipynb__ and  __LFI_train1.ipynb__ to fit models that approximate $E(Z | \theta, N)$. Datasets __data2.db__ is used in __LFI_train2.ipynb__ to fit models that approximate $E(Z | \theta, \nu, N, M)$. These functions can be used to compute confidence intervals, which for all 1-parameter problems yield exact coverage.

# In[2]:


import os, sys

# the standard module for array manipulation
import numpy as np

# the standard module for tabular data
import pandas as pd

# standard scientific python module
import scipy as sp
import scipy.stats as st

#  a function to save results
import joblib as jb

# module to reload modules
import importlib


# ### Compute $\hat{\theta}$, $t$
#    1. MLE: $\hat{\theta} = n - m$
#    2. non-MLE: $\hat{\theta} = n - m \text{ if } n > m \text{ else } 0$

# In[14]:


get_ipython().run_cell_magic('writefile', 'LFIutil.py', "\n# the standard module for array manipulation\nimport numpy as np\n\n# standard scientific python module\nimport scipy.stats as st\n\n# ------------------------\n# 1-parameter model\n# ------------------------\n\n# compute the MLE of theta\ndef theta_hat1(n):\n    return n\n\ndef Lp1(theta, n, tiny=1.e-20):\n    # compute likelihood for one or more experiments.\n    # note: st.poisson.pmf returns a numpy array if \n    # we're looping over multiple experiments, otherwise\n    # it returns a single float.\n    # be sure to handle cases in which the mean is zero\n    p1 = st.poisson.pmf(n, np.abs(theta) + tiny)\n    return p1\n\ndef t1(theta, n):\n    Ln = Lp1(theta, n)\n    Ld = Lp1(theta_hat1(n), n)\n    t  = -2*np.log(Ln / Ld)\n    return t\n\n# ------------------------\n# 2-parameter model\n# ------------------------\ndef theta_hat2(n, m, mle=True):\n    \n    # compute MLE of mean signal (the parameter of interest)\n    theta_hat = n - m\n    \n    if not mle:\n        # replace negative signal estimates by zero\n        theta_hat = theta_hat * (theta_hat > 0)\n  \n    return theta_hat\n\ndef Lp2(theta, n, m, tiny=1.e-20):\n    # compute conditional MLE of background mean\n    g  = n + m - 2 * theta\n    nu_hat = (g + np.sqrt(g*g+8*m*theta))/4\n    # compute likelihood for one or more experiments.\n    # note: st.poisson.pmf returns a numpy array if \n    # we're looping over multiple experiments, otherwise\n    # it returns a single float.\n    # be sure to handle cases in which the mean is zero\n    p1 = st.poisson.pmf(n, np.abs(theta + nu_hat) + tiny)\n    p2 = st.poisson.pmf(m, np.abs(nu_hat) + tiny)\n    return p1*p2\n\ndef t2(theta, n, m, mle=True):\n    Ln = Lp2(theta, n, m)\n    Ld = Lp2(theta_hat2(n, m, mle), n, m)\n    t  =-2*np.log(Ln / Ld)\n    return t")


# In[4]:


import LFIutil as lfi
importlib.reload(lfi);


# ### Generate datasets train.db and train1.db
# 
# $
# \begin{align}
# \theta_i & \sim \textrm{uniform}(0, 20), \\
# n_i & \sim \textrm{poisson}(\theta_i),\\
# N_i & \sim \textrm{randint}(0, 10), \textrm{ and } \\
# \textrm{ and } \\
# Z_i & = I[ t(n_i) \lt t(N_i) ],
# \end{align}
# $
# 
# where

# In[5]:


Ndata    = 510000
thetaMin =  0
thetaMax = 20
Nmin     =  0
Nmax     = 10

filename = 'data1.db'
print(filename)

theta = st.uniform.rvs(thetaMin, thetaMax, Ndata)
n     = st.poisson.rvs(theta)
N     = np.random.randint(Nmin, Nmax, Ndata)
Z1    = (n <= N).astype(np.int32)
Z2    = (lfi.t1(theta, n) < 
         lfi.t1(theta, N)).astype(np.int32)

data = pd.DataFrame({'Z1': Z1, 'Z2': Z2, 'theta': theta, 'N': N})
jb.dump(data, filename)


# ### Generate dataset 2
# 
# $
# \begin{align}
# \theta_i & \sim \textrm{uniform}(0, 20), \\
# \nu_i & \sim \textrm{uniform}(0, 20), \\
# n_i & \sim \textrm{poisson}(\theta_i + \nu_i),\\
# m_i & \sim \textrm{poisson}(\nu_i),\\
# N_i & \sim \textrm{randint}(0, 10), \\
# M_i & \sim \textrm{randint}(0, 10), \textrm{ and } \\
# \textrm{ and } \\
# Z_i & = I[ t(n_i, m_i) \lt t(N_i, M_i) ],
# \end{align}
# $

# In[6]:


nuMin    =  0
nuMax    = 20

Mmin     =  0
Mmax     = 10

filename = 'data2.csv'
print(filename)

theta = st.uniform.rvs(thetaMin, thetaMax, size=Ndata)
nu    = st.uniform.rvs(nuMin, nuMax, size=Ndata)

n     = st.poisson.rvs(theta+nu)
m     = st.poisson.rvs(nu)

N     = st.randint.rvs(Nmin, Nmax, size=Ndata)
M     = st.randint.rvs(Mmin, Mmax, size=Ndata)

MLE   = True
Z1    = (lfi.t2(theta, n, m, MLE) < 
         lfi.t2(theta, N, M, MLE)).astype(np.int32)

MLE   = False
Z2    = (lfi.t2(theta, n, m, MLE) < 
         lfi.t2(theta, N, M, MLE)).astype(np.int32)

# save in a pandas dataframe
data = pd.DataFrame({'Z1': Z1, 
                     'Z2': Z2, 
                     'theta': theta, 
                     'nu': nu, 
                     'N': N, 
                     'M': M})
# jb.dump(data, filename)
data.to_csv(filename)


# In[ ]:




