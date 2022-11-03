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


D = 1 #this does not work for large values of D





Bprime=2000000
def generate_training_data_one_parameter(Bprime, D, save_data=True):
    #T=[[theta_i],[Z_i]]
    T = [[],[]]
    for i in range(Bprime):
        #theta has to be positive because its an input to a poisson. This prior should also be close to the cound D
        theta = np.random.uniform(low=0, high=20)
        N = st.poisson.rvs(theta) #draw count samples randomly from a poisson distribution
        #this X is really N

        if D <= N:
            Z_i=1
        else:
            Z_i=0
        T[0].append(theta)
        T[1].append(Z_i)
        
    if save_data:
        Data_1_param = {'theta' : T[0], 'Z' : T[1]}
        Data_1_param = pd.DataFrame.from_dict(Data_1_param)
        halfway = len(Data_1_param)//2
        Train_data_1_param = Data_1_param.iloc[:halfway,]
        Test_data_1_param = Data_1_param.iloc[halfway:,]
        Data_1_param.to_csv('data/Uniform_Data_1_param_200k_D_eq_1.csv')
    print(Data_1_param)
    return Data_1_param
generate_training_data_one_parameter(Bprime=Bprime, D=D, save_data=True)
pass
def phat(theta):
    D=1
    count=0

    for i in range(Bprime):
        N = np.random.poisson(theta)
        if N >=D:
            count +=1
    return count/Bprime

# p_calc = sp.special.gammainc(D, theta)
Train_data_1_param, Test_data_1_param = generate_training_data_one_parameter(Bprime, D, save_data=True)

# theta = Training_data_1_param.theta
def add_simple_phat_and_p_callculated_to_test():
    phat_l = []; p_calculated=[]
    for ind, row in Train_data_1_param.iterrows():

        #print(row.theta)
        #create new column phat
        
        phat_i = phat(row.theta)
        phat_l.append(phat_i)
        p_calculated.append(sp.special.gammainc(D, row.theta))

    Train_data_1_param['phat'] = phat_l
    Train_data_1_param['p_calculated'] = p_calculated
    print(Train_data_1_param.head())
    return Train_data_1_param

Training_data_1_param=add_simple_phat_and_p_callculated_to_test()
####################################ALL MODEL AND 
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
            return outputs.flatten()
            #return final_loss / len(data_loader)

def train(optimizer, engine, early_stopping_iter, epochs):
    
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    eng = RegressionEngine(model=model, optimizer = optimizer)
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    EPOCHS=40
    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        test_loss = eng.train(test_loader)
        print("Epoch : %-10g, Training Loss: %-10g, Test Loss: %-10g" % (epoch, train_loss, test_loss))
        #print(f"{epoch}, {train_loss}, {test_loss}")
        if test_loss < best_loss:
            best_loss = test_loss

        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            #if we are not improving for 10 iterations then break the loop
            #we could save best model here
            break


# training_data_1_param  = Training_data_1_param








# we are calculating m(\theta) = \int p(Z|\theta) dZ, so 
def preprocess_and_train():
    theta = np.array(Train_data_1_param.theta)
    Z = np.array(Train_1_param.Z)

    data, targets = theta, Z
    ntargets = 1
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_targets, test_targets = train_test_split(data, 
                                                                        targets, 
                                                                        stratify=targets)
    train_targets = train_targets.reshape(-1,1)
    test_targets = test_targets.reshape(-1,1)
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

    train_dataset = CustomDataset(train_data, train_targets)
    test_dataset = CustomDataset(test_data, test_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=10, 
                                            num_workers=2, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=10, num_workers=2)

    eng = RegressionEngine(model=model, optimizer = optimizer)
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    EPOCHS=22
    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        test_loss = eng.train(test_loader)
        print("Epoch : %-10g, Training Loss: %-10g, Test Loss: %-10g" % (epoch, train_loss, test_loss))
        #print(f"{epoch}, {train_loss}, {test_loss}")
        if test_loss < best_loss:
            best_loss = test_loss

        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            #if we are not improving for 10 iterations then break the loop
            #we could save best model here
            break



# theta = np.array(Train_data_1_param.theta)
# Z = np.array(Train_data_1_param.Z)
# data, targets = theta, Z
# ntargets = 1
# from sklearn.model_selection import train_test_split
# train_data, test_data, train_targets, test_targets = train_test_split(data, 
#                                                                     targets, 
#                                                                     stratify=targets)
# train_targets = train_targets.reshape(-1,1)
# test_targets = test_targets.reshape(-1,1)
# train_data = train_data.reshape(-1,1)
# test_data = test_data.reshape(-1,1)
# train_dataset = CustomDataset(train_data, train_targets)
# test_dataset = CustomDataset(test_data, test_targets)
# train_loader = torch.utils.data.DataLoader(train_dataset, 
#                                         batch_size=10, 
#                                         num_workers=2, 
#                                         shuffle=True)

# test_loader = torch.utils.data.DataLoader(test_dataset, 
#                                         batch_size=10, num_workers=2)

# model =  RegressionModel(nfeatures=train_data.shape[1], 
#             ntargets=1,
#             nlayers=5, 
#             hidden_size=128, 
#             dropout=0.3)
# optimizer = torch.optim.Adam(model.parameters())

# train(optimizer, 
#     engine =RegressionEngine(model=model, optimizer = optimizer),
#     early_stopping_iter = 10,
#     epochs=22)

# p_calculated=[]; phat_l=[]
# from torch import Tensor
# with torch.no_grad():#SAY THETA IS UNIFOROM DIST FROM 0 TO 250
#     # new_random_theta = st.expon.rvs()
#     # new_random_theta_torch =new_random_theta.reshape(-1,1)
#     # new_random_theta_torch = torch.from_numpy(new_random_theta_torch).float()
    
#     model.eval()
#     theta_torch= np.array(Train_data_1_param.theta).reshape(-1,1)
#     theta_torch = torch.from_numpy(theta_torch).float()
#     phat = model(theta_torch)
#     for ind, row in Train_data_1_param.iterrows():

#         p_calculated.append(sp.special.gammainc(D, row.theta))

#         #phat=model(Tensor(row.theta))
#         # phat_l.append()


# Train_data_1_param['p_calculated'] = p_calculated
# # Test_data_1_param['phat'] = phat_l
# Train_data_1_param['phat'] = phat
# print(Training_data_1_param.head())










# if __name__ == '__main__':
#     generate_training_data_one_parameter(Bprime, D, save_data=True)