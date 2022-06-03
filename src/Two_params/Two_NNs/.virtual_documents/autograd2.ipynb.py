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
import joblib as jb

# update fonts
FONTSIZE = 18
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size":FONTSIZE,
})

# set a seed to ensure reproducibility
# seed = 128
# rnd  = np.random.RandomState(seed)
    
import random


x = torch.arange(8.0)
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


y = x.sum()#another function
y.backward()
x.grad


x.grad.zero_()
y = x * x
y


y_prime = y.detach()
y_prime


z = y_prime * x
z


z.sum().backward()


x.grad


y_prime


y_prime == x.grad


x.grad.zero_()
y.sum().backward()
x.grad == 2 * x


def f(a):
    return 4 * a


a = torch.randn(size=(), requires_grad=True)
a


f = f(a)
f


f.backward()
a.grad


# to double check, this is the same as f/a
f/a


def sgd(params, lr, batch_size):  
    """stochastic gradient descent: estimate the gradient of the loss with respect to some parameters params
    We then update our parameters in the direction that may reduce the loss"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size#we divide by batch_size to normalize
            param.grad.zero_()


torch.normal(0, 1, (3, 4))#example of generating data in pytorch


def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)# noise term
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features = ', features[:5], '\n\n', 'labels= ', labels[:5])


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
    print('features in one batch = ', X, '\n', 'labels in one batch= ',y)
    break


net = nn.Sequential(nn.Linear(2,1))
net


net[0].weight#0 is the first layer


net[0].bias


######################PYTORCH###############
loss_pytorch = nn.MSELoss()


#################FROM SCRATCH##############
def loss_fromscratch(f, t):
    # f and t must be of the same shape
    return  torch.mean((f - t)**2)




learning_rate = 1.e-3

optimizer_pytorch = torch.optim.SGD(net.parameters(), lr=learning_rate)
    
# optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate) 


n_epochs=3
loss = loss_pytorch
optimizer = optimizer_pytorch
for epoch in range(n_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X), y)#compare prediction to targets
        optimizer.zero_grad()#empty out the gradients
        l.backward()#calculate the gradient of the loss function 
        optimizer.step()#update the values of the parameters (weights) of model
        
    l = loss(net(features), labels)
    print(f'epoch {epoch+1}, loss {l:f}')


Bprime    = 10000
thetaMin, thetaMax =  0, 10
nuMin, nuMax = 0, 10
Mmin, Mmax =  0 , 10
Nmin, Nmax =  0,10
MLE=True
def generate_Z_lambda(Bprime, save_data=False):
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
    # print('n=', n)
    # print('m=', m)
    # print('N=', N)
    # print('M=', M)
    lambda_gen = utils.lambda_test(theta, n, m, MLE)
    # print('lambda_gen= ', lambda_gen)
    lambda_D = utils.lambda_test(theta, N, M, MLE)
    # print('lambda_D= ', lambda_D)
    #if lambda_gen <= lambda_D: Z=1, else Z=0
    Z = (lambda_gen <= lambda_D).astype(np.int32)
    

    return lambda_gen, Z


lambda_gen, Z = generate_Z_lambda(1000)

plt.scatter(lambda_gen, Z)
plt.xlabel(r'$\lambda (\theta, n, m, \hat{\nu}(\theta))$'); plt.ylabel(r'$Z$')


f, axes = plt.subplots(1, 3, sharey=True, figsize=(13,7))
axes = axes.flatten()
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
########RELU
relu = torch.relu(x)
axes[0].plot(x.detach(), relu.detach(),label=r'$ReLU(x) \equiv max(x,0)$')
#take derivative
relu.backward(torch.ones_like(x), retain_graph=True)
axes[0].plot(x.detach(), x.grad, label=r'$\frac{\partial \ ReLU(x)}{\partial x}$')
######SIGMOID
sigmoid = torch.sigmoid(x)
axes[1].plot(x.detach(), sigmoid.detach(),label=r'Sigmoid $(x)$')
# Clear out previous gradients
x.grad.data.zero_()
#take derivative
sigmoid.backward(torch.ones_like(x), retain_graph=True)
axes[1].plot(x.detach(), x.grad, label=r'$\frac{\partial \ Sigmoid(x)}{\partial x}$')
###tanh
tanh = torch.tanh(x)
axes[2].plot(x.detach(), tanh.detach(), label=r'tanh $(x)$')
x.grad.data.zero_()
tanh.backward(torch.ones_like(x),retain_graph=True)
axes[2].plot(x.detach(), x.grad, label=r'$\frac{\partial \ tanh(x)}{\partial x}$')

for ax in axes:
    ax.set_xlabel('$x$')
    ax.set_ylim(-0.2,2.5)
    ax.legend(fontsize=15)


MLE_TRUE_FILE = '../../tutorials/data1.db'
MLE_FALSE_FILE = '../../tutorials/data2.db'
#For both, Z for the two parameter problem is Z2
MLE=True

if MLE:
    datafile = MLE_TRUE_FILE
else:
    datafile = MLE_FALSE_FILE
    
data = jb.load(MLE_FALSE_FILE)
features = data[['theta', 'nu', 'N', 'M']]
target = data['Z2']
features



