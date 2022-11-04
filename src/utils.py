import numpy as np
import scipy as sp
import scipy.stats as st
import torch
from numba import njit
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

@njit
def L(n,m,theta, number_of_params=2, X=None):
    """likelihood with one or two parameter"""
    if number_of_params==1:
        return st.poisson.pmf(X, mu=theta)
    elif number_of_params==2:
        return st.poisson.pmf(n, mu=theta+nu) * st.poisson.pmf(m, mu=nu)
    

def run_sim( theta, nu):
    """Sample n ~ Pois(theta+nu), m ~ Pois(nu), and compute lambda(theta, n, m)
    return (n, m, lambda_), where each are np arrays of length chi2_expo
    """
    n = st.poisson.rvs(theta+nu, size=chi2_exp_size)
    m = st.poisson.rvs(nu, size=chi2_exp_size)
    lambda_ = lambda_test(theta, n, m, MLE)
    return (n, m, lambda_)

def run_sims(points):
    """
    input: a tuple of (theta, nu)
    Run an entire simulation (that is, generate n and m from 
    run_sim above, and calculate lambda) for each point, where 
    """
    lambda_results=[]

    for p in points:
        theta, nu = p
        n, m, lambda_ = run_sim(theta, nu)
        lambda_results.append((n, m, lambda_, theta, nu))
        print( '(theta, nu) =  (%.f, %.f) ' % (theta, nu) )
        print('\t \t with associated (n, m, lambda) = (%.f, %.f, %.f)' % (n, m, lambda_) )
    return lambda_results

@njit
def DR(s, theta):
    return sp.special.gammainc(s, theta)

@njit
def DL(s, theta):
    return 1 - sp.special.gammainc(s+1, theta)

k=1
@njit
def L_prof(n,m,theta):
    k=1
    k1 = k+1
    k2 = 0.5/k1
    g = n+m - k1*theta
    nu_hat = k2* (g+ np.sqrt(g*g +4*k1*m*theta))
    p1 = st.poisson.pmf(n, mu = theta + nu_hat)
    p2 = st.poisson.pmf(m, mu = k * nu_hat)
    
    return p1*p2

@njit
def theta_hat(n,m, MLE=True):
    theta_hat = n-m
    
    if not MLE:
        theta_hat = theta_hat * (theta_hat > 0)
    return theta_hat

@njit
def lambda_test(theta, n, m, MLE=True):
    Ln = L_prof(n,m,theta)
    Ld = L_prof(n,m, theta_hat(n,m, MLE))
    lambda_  = -2*np.log(Ln/Ld)
    return np.array(lambda_)
# import Run_Regressor_Training as TRAIN
class CustomDataset:
    """This takes the index for the data and target and gives dictionary of tensors of data and targets.
    For example we could do train_dataset = CustomDataset(train_data, train_targets); test_dataset = CustomDataset(test_data, test_targets)
 where train and test_dataset are np arrays that are reshaped to (-1,1).
 Then train_dataset[0] gives a dictionary of samples "X" and targets"""
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
        
        layers.append(nn.Sigmoid())
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
        self.model.train()
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
            return final_loss/len(data_loader)
            #return final_loss / len(data_loader)


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
