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
import utils

#Here we want to get the coverage probability from each trained regressor
#coverage probability is just the relative frequency that 
# theta \in [theta_lower, theta_upper]

D=1
MODEL_PATH='models/Regressor_D_eq_1_uniform.pth'
Bdoubleprime = 100 #size of dataset that we want to compare phat with p_calculated with
theta_vec = np.random.uniform(low=0,high=20, size=Bdoubleprime)

calculated_p_value = [sp.special.gammainc(D, theta) for theta in theta_vec]

# for i in range(Bdoubleprime):
#     with torch.no_grad():
#         model= torch.load(MODEL_PATH)
#         model.eval()

#         # training_data_1_param = pd.read_csv('data/Uniform_Data_1_param_200k_D_eq_1.csv')
#         theta_torch = np.array(training_data_1_param.theta).reshape(-1,1)
#         theta_torch = torch.from_numpy(theta_torch).float()
#         phat = model(theta_torch)

#         p_calculated=[]
#         for ind, row in training_data_1_param.iterrows():
#             p_calculated.append(sp.special.gammainc(D, row.theta))

# training_data_1_param['p_calculated'] = p_calculated
# training_data_1_param['phat'] = phat

with torch.no_grad():
    model= torch.load(MODEL_PATH)
    model.eval()
    theta_vec_torch = torch.from_numpy(theta_vec.reshape(-1,1)).float()
    phat = model(theta_vec_torch)  

phat = phat.numpy().flatten()


inference_df = pd.DataFrame({
    'theta': theta_vec,
    'calulated_p': calculated_p_value,
    'phat':phat
    
})

print(inference_df.head())
# print(phat.numpy().flatten())