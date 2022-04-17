import numpy as np
import scipy as sp
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as mp
import matplotlib.pyplot as plt
# force inline plots
get_ipython().run_line_magic("matplotlib", " inline")
plt.style.use('seaborn-deep')
import torch.nn as nn
import copy
import pandas as pd


get_ipython().run_line_magic("run", " Generate_Training_Data.ipynb")


theta, Z


training_data_1_param = pd.read_csv('data/Training_data_1_param_1M.csv')
training_data_1_param.head()


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
print(test_targets, test_data)


print(type(test_data), test_data.shape)


sc = StandardScaler()#this is always recommended for logistic regression
train_data= sc.fit_transform(train_data)
test_data = sc.transform(test_data)
train_data.mean(), (train_data.std())**2#check to make sure mean=0, std=1


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
    
train_dataset = CustomDataset(train_data, train_targets)
test_dataset = CustomDataset(test_data, test_targets)
print(train_dataset[0], train_dataset)



train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=10, 
                                           num_workers=2, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=10, num_workers=2)


# from mymodels import RegressionModel
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


model =  RegressionModel(nfeatures=train_data.shape[1], 
               ntargets=1,
               nlayers=5, 
               hidden_size=128, 
               dropout=0.3)
print(model)


# get_ipython().run_line_magic("writefile", " training/RegressionEngine.py")
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    eng = RegressionEngine(model=model, optimizer = optimizer)
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    EPOCHS=22
    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        test_loss = eng.train(test_loader)
        print("Epoch : get_ipython().run_line_magic("-10g,", " Training Loss: %-10g, Test Loss: %-10g\" % (epoch, train_loss, test_loss))")
        #print(f"{epoch}, {train_loss}, {test_loss}")
        if test_loss < best_loss:
            best_loss = test_loss

        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            #if we are not improving for 10 iterations then break the loop
            #we could save best model here
            break


optimizer = torch.optim.Adam(model.parameters())
train(optimizer, 
      engine =RegressionEngine(model=model, optimizer = optimizer),
      early_stopping_iter = 10,
      epochs=22)


from IPython.core.debugger import set_trace


def predict():
    outputs = []
    labels = []
    accuracies = []

    #evaluate
    with torch.no_grad():
        for data in test_loader:
            data_cp = copy.deepcopy(data)

            xtest = data_cp["x"]
            ytest = data_cp["y"]#y is Z values. I could add here my computed p-value for each theta,
            #and make a dataframe col1:theta, col2: Z, col3, phat, col4: computedp-value
            output = model(xtest)
            labels.append(ytest)
            outputs.append(output)

            y_predicted_cls = output.round()
            acc = y_predicted_cls.eq(ytest).sum() / float(ytest.shape[0])# number of correct predictions/sizeofytest
            #accuracies.append(acc.numpy())
            #print(f'accuracy: {acc.item():.4f}')

            del data_cp

    #     acc = y_predicted_cls.eq(ytest).sum() / float(ytest.shape[0])
    #     print(f'accuracy: {acc.item():.4f}')
            
    OUTPUTS = torch.cat(outputs).view(-1).numpy()

    LABELS = torch.cat(labels).view(-1).numpy()
    print('outputs of model: ', OUTPUTS)
    print('\nactual labels (targets Z): ', LABELS)
    return OUTPUTS.flatten(), LABELS.flatten()


OUTPUTS, LABELS = predict()


OUTPUTS.shape , LABELS.shape


plt.figure(figsize=(6,6))
plt.hist(OUTPUTS, bins=50, density=True, label = "Outputs $\hat{y}$")
plt.hist(LABELS, bins=50,density=True, label = "Labels $y$")
plt.legend()
plt.show()


print(test_data[:10])
for data in test_loader:
    print(data["x"])
    break


from torch import Tensor
D=9

def calc_phat_from_regressor(model, X):
    #calculate phat from regressor - this gives Prob(Z=1|theta)
    #here X will be input as test_data, just a 1-dimensional array of theta values (test_targets is the corresponding Z values fro them)
    X_torch = torch.from_numpy(X).float()
    
    X_torch= Tensor(X_torch)
    
    model.eval()
    phat = model(X_torch)
    phat = phat.squeeze()
    phat=phat.detach().numpy().flatten()#detaches it from the computational history/prevent future computations from being tracked

    #now analytically compute the p-value for each theta to compare with the phat
    theta = X.flatten()
    Z = test_targets.flatten()
    p_value_computed = []
#     with torch.no_grad():
#         for data in test_loader:
#             data_cp = copy.deepcopy(data)

#             theta = data_cp["x"]
#             Z = data_cp["y"]
#             #y is Z values. I could add here my computed p-value for each theta,
#             #and make a dataframe col1:theta, col2: Z, col3, phat, col4: computedp-value

#             for theta_i in theta:
#                 p_value_i = sp.special.gammainc(D, theta)
#                 p_value_computed.append(p_value_i)

#             del data_cp

    for theta_i in theta:
        p_value_i = sp.special.gammainc(D, theta_i)
        p_value_computed.append(p_value_i)

    p_value_computed=np.array(p_value_computed).flatten()
    
    phat_pcalculated_df_oneparam = pd.DataFrame({
    'theta':theta,
    'Z': Z,
    'phat':phat, 
    'p_calculated': p_value_computed})
    
    return phat_pcalculated_df_oneparam
    # return p_value_computed, phat 


p_value_computed=[]
for theta_i in test_data:
        p_value_i = sp.special.gammainc(D, theta_i)
        p_value_computed.append(p_value_i)

p_value_computed=np.array(p_value_computed).flatten()
p_value_computed.shape


phat_pcalculated_df_oneparam = calc_phat_from_regressor(model, test_data)
phat_pcalculated_df_oneparam.head()


test_data.flatten().shape


P_Z_equals_1 = calc_phat_from_regressor(model, test_data)
P_Z_equals_1


P_Z_equals_1.shape


plt.hist(P_Z_equals_1, label='$\hat{p}(Z=1|x)$')
plt.legend(fontsize=14)


plt.hist(P_Z_equals_1[P_Z_equals_1>0.5], bins=50, color='g',
            histtype='stepfilled',
            alpha=0.3,label = '$t = 1$')

plt.hist(P_Z_equals_1[P_Z_equals_1<0.5], bins=50, color='r',
            histtype='stepfilled',
            alpha=0.3,label = '$t = 0$')
plt.xlabel('$p(y=1|x)$', fontsize=13)
plt.legend()


D=9
def p_calculated():
    #for the poissons, the calculated p-value is the gamma function
    p_calculated = []


    for data in test_loader:
        data_cp = copy.deepcopy(data)

        theta = data_cp["x"]
        print(theta)
        p_value_calculated = sp.special.gammainc(D, theta)
        p_computed = np.array(p_value_calculated)
        p_calculated.append(p_computed)


        del data_cp
        

    return p_calculated


p_calc = np.array(p_calculated()).flatten(); type(p_calc)


print(len(test_data), type(test_data))
plt.figure(figsize=(10,10))
plt.hist(p_calc, label='calculated p-value', alpha=0.3)
plt.hist(P_Z_equals_1, label=r'$\hat{p}(\theta)$ from regressor', alpha=0.3)
plt.legend()


def E_hat(T):
    """The expectation value of Z as a relative frequency, this should equal p_hat, the learned parameterized distribution at a given theta"""
    num = np.array(T[1]).sum()
    den = Bprime
    return num/den



