import numpy as np
import scipy as sp
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mp
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import torch.nn as nn
import copy
import pandas as pd
import sys
# import argparse
import utils
import os
# import tqdm
import utils




# parser=argparse.ArgumentParser(description='generate training data')
# parser.add_argument('--D', type=int, help='the value of D', required=False)

# args = parser.parse_args()

# D = args.D


EPOCHS=100
# THIS NOTEBOOK TRAINS A REGRESSOR FOR ONE VALUE OF D
# training_data_2_params = pd.read_csv('data/Data_1_param'+UNIFORM_2M+'D_eq_1.csv')
TRAINING_DATA = 'TWO_PARAMETERS_TRAINING_DATA_1M.csv'
training_data_2_params = pd.read_csv(TRAINING_DATA)
input_features = ['theta', 'nu', 'N','M']

X = training_data_2_params[input_features]
Z = training_data_2_params.Z
# X, Z = np.array(X), np.array(Z)
X, Z = X.to_numpy(), Z.to_numpy()
#SHAPE OF X IS (N_SAMPLES, N_FEATURES)
# print(X, X.shape)
# print(Z, Z.shape)

data, targets = X,  Z
ntargets = 1
from sklearn.model_selection import train_test_split
train_data, test_data, train_targets, test_targets = train_test_split(data, 
                                                                      targets, 
                                                                      stratify=targets)
#Reshape the targets to have shape (something, 1)


# train_targets = train_targets.reshape(-1,1)
# test_targets = test_targets.reshape(-1,1)
# train_data = train_data.reshape(-1,1)
# test_data = test_data.reshape(-1,1)

if train_data.shape[1]  != 0:
    train_targets = train_targets.reshape(-1,1)
    test_targets = test_targets.reshape(-1,1)
else:
    train_targets = train_targets.reshape(-1,1)
    test_targets = test_targets.reshape(-1,1)
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

#DO ANOTHER CONDITION FOR TARGETS


scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


train_dataset = utils.CustomDataset(data=train_data, targets=train_targets)
test_dataset = utils.CustomDataset(data=test_data, targets=test_targets)


train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=32, 
                                           num_workers=2, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=32, num_workers=2)



model =  utils.RegressionModel(nfeatures=train_data.shape[1], 
               ntargets=1,
            #    ntargets=train_targets.shape[1],
               nlayers=5, 
               hidden_size=128, 
               dropout=0.3)


#IF YOU WANT TO TUNE HYPERPARAMETERS, USE BELOW
## params to tune: nlayers, hidden_size, dropout
# model =  utils.RegressionModel(nfeatures=train_data.shape[1], 
#                ntargets=1,
#                nlayers=params['nlayers'], 
#                hidden_size=params['hidden_size'], 
#                dropout=params['dropout'])

# %writefile training/RegressionEngine.py

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

def Run_Regressor_Training(optimizer, 
                #params,
                #engine, 
                early_stopping_iter, epochs, save_model=False):
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    eng = utils.RegressionEngine(model=model, optimizer = optimizer)
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    EPOCHS=15
    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        test_loss = eng.evaluate(test_loader)
        print("Epoch : %-10g, Training Loss: %-10g, Test Loss: %-10g" % (epoch, train_loss, test_loss))
        #print(f"{epoch}, {train_loss}, {test_loss}")
        if test_loss < best_loss:
            best_loss = test_loss
            if save_model:
                # torch.save(model.state_dict(), "models/Regressor_D_eq_1_uniform.pth")
                # torch.save(model.state_dict() , 'TWO_PARAMETERS_TRAINED _MODEL.pth')
                torch.save(model , 'TWO_PARAMETERS_TRAINED _MODEL.pth')

        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            #if we are not improving for 10 iterations then break the loop
            #we could save best model here
            break
    

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
#         temp_loss = Run_Regressor_Training(i, params, save_model=False)#we don't want to save the model when it is still being tuned
#         all_losses.append(temp_loss)



if __name__ == "__main__":
    Run_Regressor_Training(optimizer=optimizer,
    #engine=eng
      early_stopping_iter = 15,
      epochs=EPOCHS,
      save_model=True)
    # print('saved this trained model as model')

