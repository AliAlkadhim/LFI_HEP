import numpy as np
import scipy as sp
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as mp
import matplotlib.pyplot as plt
# force inline plots
plt.style.use('seaborn-deep')
import torch.nn as nn
import copy
import pandas as pd
import optuna
# def L(n,m,theta, number_of_params=2, X=None):
#     """likelihood with one or two parameter"""
#     if number_of_params==1:
#         return st.poisson.pmf(X, mu=theta)
#     elif number_of_params==2:
#         return st.poisson.pmf(n, mu=theta+nu) * st.poisson.pmf(m, mu=nu)
    

# import Run_Regressor_Training as TRAIN
class CustomDataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets=targets
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        
        current_sample = self.data[idx, :]
        current_target = self.targets[idx]
        return {"x": torch.tensor(current_sample, dtype = torch.float),
               "y": torch.tensor(current_target, dtype= torch.float),
               }#this already makes the targets made of one tensor (of one value) each
    


class RegressionModel(nn.Module):
    #inherit from the super classdddddddddddd
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size, dropout):
        #nlayers, hidden_size, dropout are model parameters that could be tuned
        super().__init__()
        layers = []
        for _ in range(nlayers):
            if len(layers) ==0:
                #inital layer has to have size of input features as its input layer
                #its output layer can have any size but it must match the size of the input layer of the next linear layer
                #here we choose its output layer as the hidden size (fully connected)
                layers.append(nn.Linear(nfeatures, hidden_size))
                #batch normalization
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                #ReLU activation 
                layers.append(nn.ReLU())
            else:
                #if this is not the first layer (we dont have layers)
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
                #output layer:
        layers.append(nn.Linear(hidden_size, ntargets)) 
        
        # layers.append(nn.Sigmoid())
            #we have defined sequential model using the layers in oulist 
        self.model = nn.Sequential(*layers)
            
    
    def forward(self, x):
        return self.model(x)


class RegressionEngine:
    """loss, training and evaluation"""
    def __init__(self, model, optimizer):
                 #, device):
        self.model = model
        #self.device= device
        self.optimizer = optimizer
        
    #the loss function returns the loss function. It is a static method so it doesn't need self
    @staticmethod
    def loss_fun(targets, outputs):
        return nn.MSELoss()(outputs, targets)
        # return nn.nn.NLLLoss()(outputs, targets)
        # return nn.KLDivLoss()(outputs, targets)


    def train(self, data_loader):
        """the training function: takes the training dataloader"""
        self.model.train()#put it in train mode
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()#only optimize weights for the current batch, otherwise it's meaningless!
            inputs = data["x"]
            targets = data["y"]
            outputs = self.model(inputs)
            loss = self.loss_fun(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    
    def evaluate(self, data_loader):
        """the training function: takes the training dataloader"""
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            inputs = data["x"]#.to(self.device)
            targets = data["y"]#.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fun(targets, outputs)
            final_loss += loss.item()
        return final_loss / len(data_loader) # outputs.flatten()


#Objective function for tuning with optuna
# nlayers, hidden_size, dropout
# model =  utils.RegressionModel(nfeatures=train_data.shape[1], 
#                ntargets=1,
#                nlayers=params['nlayers'], 
#                hidden_size=params['hidden_size'], 
#                dropout=params['dropout'])
# def objective(trial):
#     """#Objective function for tuning with optuna. params is a dictionary of parameters that we want to tune
#     This dictionary is called everytime a trial is started. The key is the number of the parameter name in you model.
#     The value name could be different name (but why choose a different name?)

#         """
#     params = {
#         "nlayers": trial.suggest_int("nlayers", 1, 10), #number of layers could be between 1 and 7
#         "hidden_size" : trial.suggest_int("hidden_size", 16, 2048), #number of
#          "dropout" : trial.suggest_uniform("dropout", 0.1, 0.7),#sample the dropout (which will always be a fraction from a uniform[0.1,0.7]
#     }
#     all_losses =[]
#     for i in range(5):
#         temp_loss = TRAIN.Run_Regressor_Training(i, params, save_model=False)#we don't want to save the model when it is still being tuned
#         all_losses.append(temp_loss)


#     return 


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
    

