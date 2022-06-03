import os, sys

# the standard module for array manipulation
import numpy as np

# the standard module for tabular data
import pandas as pd

# standard scientific python module
import scipy as sp
import scipy.stats as st


import LFIutil as lfi

##################GENERATE DATA FOR 2 PARAMETER MODEL########


nuMin    =  0
nuMax    = 20

Mmin     =  0
Mmax     = 10

Nmin =0
Nmax=10

thetaMin =  0
thetaMax = 20

Ndata    = 510000

theta = st.uniform.rvs(thetaMin, thetaMax, size=Ndata)
nu    = st.uniform.rvs(nuMin, nuMax, size=Ndata)

n     = st.poisson.rvs(theta+nu)
m     = st.poisson.rvs(nu)

N     = st.randint.rvs(Nmin, Nmax, size=Ndata)
M     = st.randint.rvs(Mmin, Mmax, size=Ndata)

def theta_hat2(n, m, mle=True):
    
    # compute MLE of mean signal (the parameter of interest)
    theta_hat = n - m
    
    if not mle:
        # replace negative signal estimates by zero
        theta_hat = theta_hat * (theta_hat > 0)
  
    return theta_hat

theta_hat_MLE = theta_hat2(n, m, mle=True)
theta_hat_nonMLE = theta_hat2(n, m, mle=False)


Z_MLE_TRUE = (lfi.t2(theta, n, m, MLE=True) < 
         lfi.t2(theta, N, M, MLE=True)).astype(np.int32)

Z_MLE_FALSE = (lfi.t2(theta, n, m, MLE=False) < 
         lfi.t2(theta, N, M, MLE=False)).astype(np.int32)

data = pd.DataFrame({'Z_MLE_TRUE': Z_MLE_TRUE, 
                     'Z_MLE_FALSE': Z_MLE_FALSE, 
                     'theta': theta, 
                     'theta_hat_MLE': theta_hat_MLE,
                     'theta_hat_nonMLE': theta_hat_nonMLE,
                     'nu': nu, 
                     'N': N, 
                     'M': M})

DATA_FILENAME = 'DATA_FOR_TWO_NNs.csv'
data.to_csv(DATA_FILENAME)