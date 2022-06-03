
# the standard module for array manipulation
import numpy as np

# standard scientific python module
import scipy.stats as st

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import matplotlib as mp
seed = 128
rnd  = np.random.RandomState(seed)

FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set usetex = False if LaTex is not 
# available on your system or if the 
# rendering is too slow
mp.rc('text', usetex=True)



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

def sim_nm_lambdas(theta, nu):
    """Sample n ~ Pois(theta+nu), m ~ Pois(nu), and compute lambda_gen(theta, n, m) and lambda_D(theta, N, M)
    return (n, m, lambda_gen, lambda_D)"""
    n = st.poisson.rvs(theta+nu, size=Bprime)
    m = st.poisson.rvs(nu, size=Bprime)
    lambda_gen = t2(theta, n, m, MLE)
    lambda_D = t2(theta, N, M, MLE)
    return (n, m, lambda_gen, lambda_D)


def split_t_x(df, target, source):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    t = np.array(df[target])
    x = np.array(df[source])
    return t, x

def get_batch(x, t, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = rnd.choice(len(x), batch_size)
    batch_x = x[rows]
    batch_t = t[rows]
    return (batch_x, batch_t)


def average_loss(f, t):
    # f and t must be of the same shape
    return  torch.mean((f - t)**2)

def validate(model, avloss, inputs, targets):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode
    
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        x = torch.from_numpy(inputs).float()
        t = torch.from_numpy(targets).float()
        # remember to reshape!
        o = model(x).reshape(t.shape)
    return avloss(o, t)

###############TRAIN
def train(model, optimizer, avloss, getbatch,
          train_x, train_t, 
          valid_x, valid_t,
          batch_size, 
          n_iterations, traces, 
          step=50):
    
    # to keep track of average losses
    #traces = xx, training loss, validation loss
    xx, yy_t, yy_v = traces
    
    n = len(valid_x)
    
    print('Iteration vs average loss')
    print("%10s\t%10s\t%10s" % \
          ('iteration', 'train-set', 'valid-set'))
    
    for ii in range(n_iterations):

        # set mode to training so that training specific 
        # operations such as dropout are enabled.
        model.train()
        
        # get a random sample (a batch) of data (as numpy arrays)
        batch_x, batch_t = getbatch(train_x, train_t, batch_size)
        
        # convert the numpy arrays batch_x and batch_t to tensor 
        # types. The PyTorch tensor type is the magic that permits 
        # automatic differentiation with respect to parameters. 
        # However, since we do not need to take the derivatives
        # with respect to x and t, we disable this feature
        with torch.no_grad(): # no need to compute gradients 
            # wrt. x and t
            x = torch.from_numpy(batch_x).float()
            t = torch.from_numpy(batch_t).float()      

        # compute the output of the model for the batch of data x
        # Note: outputs is 
        #   of shape (-1, 1), but the tensor targets, t, is
        #   of shape (-1,)
        # In order for the tensor operations with outputs and t
        # to work correctly, it is necessary that they have the
        # same shape. We can do this with the reshape method.
        outputs = model(x).reshape(t.shape)
   
        # compute a noisy approximation to the average loss
        empirical_risk = avloss(outputs, t)
        
        # use automatic differentiation to compute a 
        # noisy approximation of the local gradient
        optimizer.zero_grad()       # clear previous gradients
        empirical_risk.backward()   # compute gradients of the loss (wrt weights)
        
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer.step()            # move one step
        
        if ii % step == 0:
            
            acc_t = validate(model, avloss, train_x[:n], train_t[:n]) 
            acc_v = validate(model, avloss, valid_x[:n], valid_t[:n])

            if len(xx) < 1:
                xx.append(0)
                print("%10d\t%10.6f\t%10.6f" % \
                      (xx[-1], acc_t, acc_v))
            else:
                xx.append(xx[-1] + step)
                print("\r%10d\t%10.6f\t%10.6f" % \
                      (xx[-1], acc_t, acc_v), end='')
                
            yy_t.append(acc_t)
            yy_v.append(acc_v)
    print()      
    return (xx, yy_t, yy_v)


###
def plot_average_loss(traces):
    
    xx, yy_t, yy_v = traces
    
    # create an empty figure
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)

    ax.set_title("Average loss")
    
    ax.plot(xx, yy_t, 'b', lw=2, label='Training')
    ax.plot(xx, yy_v, 'r', lw=2, label='Validation')

    ax.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax.set_ylabel('average loss', fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')
    ax.legend(loc='upper right')

    plt.show()


class Model(nn.Module):
    
    def __init__(self, n_inputs=4, n_nodes=20, n_layers=5):

        # call constructor of base (or super, or parent) class
        super(Model, self).__init__()

        self.layers = []
        
        # create input layer
        self.layer0 = nn.Linear(n_inputs, n_nodes)
        self.layers.append(self.layer0)

        # create "hidden" layers
        for l in range(1, n_layers):
            cmd = 'self.layer%d = nn.Linear(%d, %d)' % \
            (l, n_nodes, n_nodes)
            exec(cmd)
            cmd = 'self.layers.append(self.layer%d)' % l
            exec(cmd)
          
        # create output layer
        cmd = 'self.layer%d = nn.Linear(%d, 1)' % (n_layers, n_nodes)
        exec(cmd)
        cmd = 'self.layers.append(self.layer%d)' % n_layers
        exec(cmd)

    # define (required) method to compute output of network
    def forward(self, x):
        y = x
        for layer in self.layers[:-1]:
            y = layer(y)
            y = torch.relu(y)
        y = self.layers[-1](y)
        y = torch.sigmoid(y)
        return y

##############
XMIN  = 0
XMAX  = 20
XBINS = 200
NU    = 3
D     = [(1, 0), (2, 0), (3, 0), 
         (1, 1), (2, 1), (3, 1)]
MLE=True
if MLE:
    target = 'Z_MLE_TRUE'
    WHICH  = 'MLE'
else:
    target = 'Z_MLE_FALSE   '
    WHICH  = 'nonMLE'
    
source = ['theta', 'nu', 'N', 'M']
def hist_data(nu, N, M,
              xbins=XBINS,
              xmin=XMIN, 
              xmax=XMAX,
              mle=MLE,
              Ndata=100000):

    theta = st.uniform.rvs(xmin, xmax, size=Ndata)
    n = st.poisson.rvs(theta + nu)
    m = st.poisson.rvs(nu, size=Ndata)
    Z = (t2(theta, n, m, MLE) < 
         t2(theta, N, M, MLE)).astype(np.int32)

    xrange = (xmin, xmax)

    # weighted histogram   (count the number of ones per bin)
    y1, bb = np.histogram(theta, 
                          bins=xbins, 
                          range=xrange, 
                          weights=Z)
    
    # unweighted histogram (count number of ones and zeros per bin)
    yt, _ = np.histogram(theta, 
                         bins=xbins, 
                         range=xrange)

    y =  y1 / yt    
    
    return y, bb

def plot_data(nu, D, 
              func=None,
              xbins=XBINS,
              xmin=XMIN, 
              xmax=XMAX,
              gfile='fig_data2_%s.png' % WHICH, 
              fgsize=(10, 6)):
    # make room for 6 sub-plots
    fig, ax = plt.subplots(nrows=2, 
                           ncols=3, 
                           figsize=fgsize)
    # padding
    plt.subplots_adjust(hspace=0.01); plt.subplots_adjust(wspace=0.20)
    # use flatten() to convert a numpy array of shape (nrows, ncols) to a 1-d array. 
    ax = ax.flatten()
    for j, (N, M) in enumerate(D):    
        #y is the histogram of theta (weighted/unweighted) and bb are its bin edges
        y, bb = hist_data(nu, N, M)
        #hist data ON THE FLY GENERATES samples of (theta, n, m) and calculates Z for those, and
        #returns (yy,bb) above.
        ax[j].set_xlim(xmin, xmax)
        ax[j].set_ylim(0, 1)
        ax[j].set_xlabel(r'$\theta$', fontsize=FONTSIZE)
        ax[j].set_ylabel(r'$E(Z|\theta, \nu)$', fontsize=FONTSIZE)
        
        x = (bb[1:]+bb[:-1])/2
        ax[j].plot(x, y, 'b', lw=2, label='weighted/unweighted histogrammed approx')
        
        if func:
            p, _ = func(nu, N, M)
            ax[j].plot(x, p, 'r', lw=2, label='model')
        
        ax[j].grid(True, which="both", linestyle='-')
        ax[j].text(10.1, 0.42, r'$N, M = %d, %d$' % (N, M), fontsize=FONTSIZE) 

        ax[j].text(10.1, 0.30, r'$\nu = %5.1f$' % nu, fontsize=FONTSIZE) 

        ax[j].legend(loc='upper right')    
    # hide unused sub-plots
    for k in range(j+1, len(ax)):
        ax[k].set_visible(False)
    plt.tight_layout()
    # plt.savefig(gfile)
    plt.show()