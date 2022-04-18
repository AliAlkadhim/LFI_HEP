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
import sys
import argparse

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


##############################################################################

D_lower, D_upper = 0, 13
#load training data
for D in range(D_lower, D_upper):
    training_data_1_param = pd.read_csv('data/Training_data_1_param_100k_D_%d.csv' % D)
    theta = np.array(training_data_1_param.theta)
    Z = np.array(training_data_1_param.Z)
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

    model =  RegressionModel(nfeatures=train_data.shape[1], 
               ntargets=1,
               nlayers=5, 
               hidden_size=128, 
               dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters())

    train(optimizer, 
      engine =RegressionEngine(model=model, optimizer = optimizer),
      early_stopping_iter = 10,
      epochs=22)
    #save model
    models_base = 'models/1_param_D_0_20'
    torch.save(model.state_dict(), models_base + 'model_D_%d.pt' % D)
    print('saved this trained model as model_D_%d.pt' % D)