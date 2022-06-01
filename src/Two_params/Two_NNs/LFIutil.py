
# the standard module for array manipulation
import numpy as np

# standard scientific python module
import scipy.stats as st

# ------------------------
# 1-parameter model
# ------------------------

# compute the MLE of theta
def theta_hat1(n):
    return n

def Lp1(theta, n, tiny=1.e-20):
    # compute likelihood for one or more experiments.
    # note: st.poisson.pmf returns a numpy array if 
    # we're looping over multiple experiments, otherwise
    # it returns a single float.
    # be sure to handle cases in which the mean is zero
    p1 = st.poisson.pmf(n, np.abs(theta) + tiny)
    return p1

def t1(theta, n):
    Ln = Lp1(theta, n)
    Ld = Lp1(theta_hat1(n), n)
    t  = -2*np.log(Ln / Ld)
    return t

# ------------------------
# 2-parameter model
# ------------------------
def theta_hat2(n, m, MLE=True):
    
    # compute MLE of mean signal (the parameter of interest)
    theta_hat = n - m
    
    if not MLE:
        # replace negative signal estimates by zero
        theta_hat = theta_hat * (theta_hat > 0)
  
    return theta_hat

def Lp2(theta, n, m, tiny=1.e-20):
    # compute conditional MLE of background mean
    g  = n + m - 2 * theta
    nu_hat = (g + np.sqrt(g*g+8*m*theta))/4
    # compute likelihood for one or more experiments.
    # note: st.poisson.pmf returns a numpy array if 
    # we're looping over multiple experiments, otherwise
    # it returns a single float.
    # be sure to handle cases in which the mean is zero
    p1 = st.poisson.pmf(n, np.abs(theta + nu_hat) + tiny)
    p2 = st.poisson.pmf(m, np.abs(nu_hat) + tiny)
    return p1*p2

def t2(theta, n, m, MLE):
    Ln = Lp2(theta, n, m)
    Ld = Lp2(theta_hat2(n, m, MLE), n, m)
    t  =-2*np.log(Ln / Ld)
    return t
