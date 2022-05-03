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


Ndata    = 510000
thetaMin =  0
thetaMax = 20
Dmin     =  0
Dmax     = 10

filename = 'data.db'
print(filename)

D     = np.random.randint(Dmin, Dmax, Ndata)
theta = st.uniform.rvs(thetaMin, thetaMax, Ndata)
N     = st.poisson.rvs(theta)
Z     = (N <= D).astype(np.int32)

data = pd.DataFrame({'Z': Z, 'theta': theta, 'D': D})
jb.dump(data, filename)



