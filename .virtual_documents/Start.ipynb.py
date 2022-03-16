import numpy as np
import scipy as sp
import scipy.stats as st





Bprime=10000
D = 10
L_obs=30 
#b= mean background
print('The size of $B\'$: ', Bprime)
print('The observed signal signal $D$ or $N$: ', D)
print('The observed luminosity: ', L_obs)
# print('The observed background'


def lambd(D, theta, thetahat=1):#test statistic
    L_num = st.norm.pdf(D, loc= theta, scale=1)#the gaussian pdf of D counts
    return L_num


#T=[[theta_i],[Z_i]]
T = [[],[]]
for i in range(Bprime):
    theta = round(np.random.normal(10))#draw a count theta from a radom poisson prior, it has to be count because its an input to a poisson. This prior should also be close to the cound D
    X_mean = np.random.poisson(lam=theta) #draw count samples randomly from a poisson distribution
    lam_true = lambd(D, theta)
    lam_i = lambd(X_mean, theta)
    if lam_i < lam_true:
        Z_i=1
    else:
        Z_i=0
    T[0].append(theta)
    T[1].append(Z_i)


def E_hat(T):
    """The expectation value of Z as a relative frequency, this should equal p_hat, the learned parameterized distribution at a given theta"""
    num = np.array(T[1]).sum()
    den = Bprime
    return num/den


E_hat(T)


def p_calculated(theta):
    return sp.special.gammainc(D, theta)


p = p_calculated(theta = round(np.random.normal(10))); p


np.random.gamma(5)












