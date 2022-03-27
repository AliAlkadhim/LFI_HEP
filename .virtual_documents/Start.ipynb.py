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





Bprime=1000000
D = 100
L_obs=30 
#b= mean background
print('The size of B: ', Bprime)
print('The observed signal signal N (or bold X in the paper): ', D)
print('The observed luminosity: ', L_obs)
# print('The observed background'


def L(D, theta):
    return st.poisson.pmf(D, mu=theta)
    
def labd_one_param(D, theta):
    num = L(D, theta)
    den = L(D, D)
    return -2 * np.log(num/den)


#T=[[theta_i],[Z_i]]
T = [[],[]]
for i in range(Bprime):
    theta = st.expon.rvs() #sample theta from an exponential distribution
    #theta has to be positive because its an input to a poisson. This prior should also be close to the cound D
   
    X = np.random.poisson(lam=theta) #draw count samples randomly from a poisson distribution
    lam_true = labd_one_param(D, theta)
    lam_i = labd_one_param(X, theta)
    if lam_i > lam_true:
        Z_i=1
    else:
        Z_i=0
    T[0].append(theta)
    T[1].append(Z_i)


def E_hat(T):
    """The expectation value of Z as a relative frequency, this should equal p_hat, the learned parameterized distribution at a given theta"""
    num = np.array(T[1]).sum()
    den = Bprime
    return num/den


E_hat(T)


def p_calculated(theta):
    return sp.special.gammainc(D, theta)


p = p_calculated(theta = round(np.random.normal(10))); p


np.array(T[1]).sum()


data, targets = np.array(T[0]), np.array(T[1])


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


sc = StandardScaler()#this is always recommended for logistic regression
train_data= sc.fit_transform(train_data)
test_data = sc.transform(test_data)
train_data.mean(), (train_data.std())**2#check to make sure mean=0, std=1


import mymodels #this library can be found at https://github.com/AliAlkadhim/MyMLFramework/blob/main/mymodels.py
mymodels.CustomDataset.__doc__


from mymodels import CustomDataset
train_dataset = CustomDataset(train_data, train_targets)
test_dataset = CustomDataset(test_data, test_targets)
train_dataset[0]



train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=10, 
                                           num_workers=2, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=10, num_workers=2)


# from mymodels import RegressionModel
class RegressionModel(nn.Module):
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
            self.optimizer.zero_grad()
            inputs = data["x"]#.to(self.device)
            targets = data["y"]#.to(self.device)
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
            return outputs
            #return final_loss / len(data_loader)


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


outputs = []
labels = []
accuracies = []

#evaluate
with torch.no_grad():
    for data in test_loader:
        data_cp = copy.deepcopy(data)
        
        xtest = data_cp["x"]
        ytest = data_cp["y"]
        output = model(xtest)
        labels.append(ytest)
        outputs.append(output)
        
        y_predicted_cls = output.round()
        acc = y_predicted_cls.eq(ytest).sum() / float(ytest.shape[0])#bumber of correct predictions/sizeofytest
        #accuracies.append(acc.numpy())
        #print(f'accuracy: {acc.item():.4f}')
        
        del data_cp

#     acc = y_predicted_cls.eq(ytest).sum() / float(ytest.shape[0])
#     print(f'accuracy: {acc.item():.4f}')
outputs[0:2]


OUTPUTS = torch.cat(outputs).view(-1).numpy()

LABELS = torch.cat(labels).view(-1).numpy()
print(OUTPUTS, LABELS)
plt.rcParams["figure.figsize"] = (6,6)
plt.hist(OUTPUTS, bins=50, density=True, label = "Outputs $\hat{y}$")
plt.hist(LABELS, bins=50,density=True, label = "Labels $y$")
plt.legend()
plt.show()


def compute_prob(model, xx):
    # convert from numpy array to a torch tensor of type float
    """Gives P(y=1|x)"""
    x = torch.from_numpy(xx).float()#.to(device)

    # compute p(1|x)
    model.eval() # evaluation mode
    p = model(x)#.to(device)

    # squeeze() removes extraneous dimensions
    p = p.squeeze()

    # detach().numpy() converts back to a numpy array
    p = p.detach().cpu().numpy()
    return p
p = compute_prob(model, test_data)
print(p)
plt.hist(p[p>0.5], bins=50, color='g',
            histtype='stepfilled',
            alpha=0.3,label = '$t = 1$')
plt.hist(p[p<0.5], bins=50, color='r',
            histtype='stepfilled',
            alpha=0.3,label = '$t = 0$')
plt.xlabel('$p(y=1|x)$', fontsize=13)
plt.legend()


def Algorithm2(D=2, theta_0):
    
    
    
    return actual_p_value, regressed_p_value
