import numpy as np
import scipy as sp
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mp
import matplotlib.pyplot as plt
# force inline plots
plt.style.use('seaborn-deep')
import torch.nn as nn
import copy
import pandas as pd
import sys
import argparse
import utils
import os

TRAINING_DATA = os.environ.get("TRAINING_DATA")
RUN_NAME = os.environ.get("RUN_NAME")


EPOCHS=15
# THIS NOTEBOOK TRAINS A REGRESSOR FOR ONE VALUE OF D
# training_data_1_param = pd.read_csv('data/Data_1_param'+UNIFORM_2M+'D_eq_1.csv')
training_data_1_param = pd.read_csv('data/'+RUN_NAME+'D_eq_1.csv')
theta = np.array(training_data_1_param.theta)
Z = np.array(training_data_1_param.Z)

data, targets = theta, Z
ntargets = 1
from sklearn.model_selection import train_test_split
train_data, test_data, train_targets, test_targets = train_test_split(data, 
                                                                      targets, 
                                                                      stratify=targets)
#Reshape the targets to have shape (something, 1)

train_targets = train_targets.reshape(-1,1)
test_targets = test_targets.reshape(-1,1)
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

    
train_dataset = utils.CustomDataset(train_data, train_targets)
test_dataset = utils.CustomDataset(test_data, test_targets)



train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=10, 
                                           num_workers=2, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=10, num_workers=2)



model =  utils.RegressionModel(nfeatures=train_data.shape[1], 
               ntargets=1,
               nlayers=5, 
               hidden_size=128, 
               dropout=0.3)


# %writefile training/RegressionEngine.py

optimizer = torch.optim.Adam(model.parameters())

def Run_Regressor_Training(optimizer, 
                #engine, 
                early_stopping_iter, epochs, save_model=False):
    
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    eng = utils.RegressionEngine(model=model, optimizer = optimizer)
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    EPOCHS=15
    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        test_loss = eng.train(test_loader)
        print("Epoch : %-10g, Training Loss: %-10g, Test Loss: %-10g" % (epoch, train_loss, test_loss))
        #print(f"{epoch}, {train_loss}, {test_loss}")
        if test_loss < best_loss:
            best_loss = test_loss
            if save_model:
                # torch.save(model.state_dict(), "models/Regressor_D_eq_1_uniform.pth")
                torch.save(model, "models/Regressor_D_eq_1_uniform.pth")

        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            #if we are not improving for 10 iterations then break the loop
            #we could save best model here
            break
    
    return best_loss




if __name__ == "__main__":
    Run_Regressor_Training(optimizer=optimizer,
    #engine=eng
      early_stopping_iter = 15,
      epochs=EPOCHS,
      save_model=True)
    print('saved this trained model as model')

