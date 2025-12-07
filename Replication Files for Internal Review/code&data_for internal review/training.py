# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.utils.data as Data
import copy
import random
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
class NN(torch.nn.Module):
    def __init__(self,input_size,output_size,neurons,dropout_rate=0.1):
        super(NN,self).__init__()    
               
        layers = []
        in_dim = input_size
        for hidden_size in neurons:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  
            in_dim = hidden_size

        layers.append(nn.Linear(neurons[-1], output_size))
        layers.append(torch.nn.Sigmoid())# output_layer

        self.net = nn.Sequential(*layers)     
        
    def forward(self,x):
        prediction = self.net(x)                               
        return prediction

class NN2(torch.nn.Module):
    def __init__(self,input_size1,input_size2,output_size,neurons1,neurons2,dropout_rate=0.1):
        super(NN2,self).__init__()
        self.batchnorm = torch.nn.BatchNorm1d(neurons1[-1]+neurons2[-1])
        self.outputlayer = torch.nn.Sequential(
                            torch.nn.Dropout(dropout_rate),
                            torch.nn.Linear(neurons1[-1]+neurons2[-1],output_size),
                            torch.nn.Sigmoid()
                            )
        layers = []
        in_dim = input_size1
        for hidden_size in neurons1:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  
            in_dim = hidden_size
        self.net1 = nn.Sequential(*layers) 
        
        layers = []
        in_dim = input_size2
        for hidden_size in neurons2:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  
            in_dim = hidden_size
        self.net2 = nn.Sequential(*layers)  

    def forward(self,data1,data2):
        x = torch.hstack((self.net1(data1),self.net2(data2)))
        x = self.batchnorm(x)
        prediction = self.outputlayer(x)                               
        return prediction

class NN22(torch.nn.Module):
    def __init__(self,input_size1,input_size2,output_size,neurons1,neurons2,neurons,dropout_rate=0.1):
        super(NN22,self).__init__()
        self.batchnorm = torch.nn.BatchNorm1d(neurons1[-1]+neurons2[-1])
        self.outputlayer = torch.nn.Sequential(
                            torch.nn.Dropout(dropout_rate),
                            torch.nn.Linear(neurons1[-1]+neurons2[-1],neurons[0]),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                            torch.nn.Linear(neurons[0],output_size),
                            torch.nn.Sigmoid()
                            )

        layers = []
        in_dim = input_size1
        for hidden_size in neurons1:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  
            in_dim = hidden_size
        self.net1 = nn.Sequential(*layers)  
        
        layers = []
        in_dim = input_size2
        for hidden_size in neurons2:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  
            in_dim = hidden_size
        self.net2 = nn.Sequential(*layers) 

    def forward(self,data1,data2):
        x = torch.hstack((self.net1(data1),self.net2(data2)))
        x = self.batchnorm(x)
        prediction = self.outputlayer(x)                               
        return prediction

class NN3(torch.nn.Module):
    def __init__(self,input_size1,input_size2,input_size3,output_size,neurons1,neurons2,neurons3,dropout_rate=0.1):
        super(NN3,self).__init__()
        self.batchnorm = torch.nn.BatchNorm1d(neurons1[-1]+neurons2[-1]+neurons3[-1])
        self.outputlayer = torch.nn.Sequential(
                            torch.nn.Dropout(dropout_rate),
                            torch.nn.Linear(neurons1[-1]+neurons2[-1]+neurons3[-1],output_size),
                            torch.nn.Sigmoid()
                            )
        layers = []
        in_dim = input_size1
        for hidden_size in neurons1:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  
            in_dim = hidden_size
        self.net1 = nn.Sequential(*layers)  
        
        layers = []
        in_dim = input_size2
        for hidden_size in neurons2:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) 
            in_dim = hidden_size
        self.net2 = nn.Sequential(*layers)  
        layers = []
        in_dim = input_size3
        for hidden_size in neurons3:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) 
            in_dim = hidden_size
        self.net3 = nn.Sequential(*layers)

    def forward(self,data1,data2,data3):
        x = torch.hstack((self.net1(data1),self.net2(data2),self.net3(data3)))
        x = self.batchnorm(x)
        prediction = self.outputlayer(x)                               
        return prediction
    
class NN32(torch.nn.Module):
    def __init__(self,input_size1,input_size2,input_size3,output_size,neurons1,neurons2,neurons3,neurons,dropout_rate=0.1):
        super(NN32,self).__init__()
        self.batchnorm = torch.nn.BatchNorm1d(neurons1[-1]+neurons2[-1]+neurons3[-1])
        self.outputlayer = torch.nn.Sequential(
                            torch.nn.Dropout(dropout_rate),
                            torch.nn.Linear(neurons1[-1]+neurons2[-1]+neurons3[-1],neurons[0]),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                            torch.nn.Linear(neurons[0],output_size),
                            torch.nn.Sigmoid()
                            )

        layers = []
        in_dim = input_size1
        for hidden_size in neurons1:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) 
            in_dim = hidden_size
        self.net1 = nn.Sequential(*layers) 
        
        layers = []
        in_dim = input_size2
        for hidden_size in neurons2:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) 
            in_dim = hidden_size
        self.net2 = nn.Sequential(*layers)  

        layers = []
        in_dim = input_size3
        for hidden_size in neurons3:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) 
            in_dim = hidden_size
        self.net3 = nn.Sequential(*layers) 
        

    def forward(self,data1,data2,data3):
        x = torch.hstack((self.net1(data1),self.net2(data2),self.net3(data3)))
        x = self.batchnorm(x)
        prediction = self.outputlayer(x)                               
        return prediction



def FFN(train_data,test_data,neurons,datatype,year,predict_label,ensemble=10):
    valid_data = train_data.loc[train_data['year']==(year-1)]
    train_data = train_data.loc[train_data['year']<(year-1)]
    train_x = torch.FloatTensor(train_data.iloc[:,3:-2].values)
    train_y = torch.FloatTensor(train_data.iloc[:,-2].values.reshape(-1,1))
    train_sample_weight = torch.FloatTensor(train_data.iloc[:,-1].values.reshape(-1,1))
    valid_x = torch.FloatTensor(valid_data.iloc[:,3:-2].values)
    valid_y = torch.FloatTensor(valid_data.iloc[:,-2].values.reshape(-1,1))
    valid_sample_weight =  torch.FloatTensor(valid_data.iloc[:,-1].values.reshape(-1,1))
    for i in range(ensemble):
        setup_seed(i)
        dataset = Data.TensorDataset(train_x,train_y,train_sample_weight)
        train_loader = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=True) 
        
        Net = NN(train_x.shape[1],1,neurons)
        optimizer = torch.optim.Adam(Net.parameters())
        
        select_model = None
        error = np.inf
        j = 0
        for k in range(1000):       
            for step,(batch_x,batch_y,sample_weight) in enumerate(train_loader): 
                if batch_x.shape[0] < 64:
                    pass
                else:
                    prediction = Net(batch_x)
                    loss_func = torch.nn.BCELoss(weight=sample_weight)
                    loss = loss_func(prediction,batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            j += 1
            Net.eval()
            predict_y = Net(valid_x)
            loss_func = torch.nn.BCELoss(weight=valid_sample_weight)
            new_error = loss_func(predict_y,valid_y).data.numpy()
            Net.train()
            # print(new_error)
            if new_error < error:
                error = new_error
                select_model = copy.deepcopy(Net)
                j = 0
            if j> 50:
                break
        select_model.eval()
        with open(r'models_final\FFN'+str(neurons)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','wb') as f:
            pickle.dump(select_model,f)


def logistic(data,datatype,year,predict_label):
    model = LogisticRegression(penalty='none')
    model.fit(data.iloc[:,3:-2],data.iloc[:,-2],sample_weight=data.iloc[:,-1]) 
    with open(r'models_final\logit'+'_'+datatype+'_'+str(year)+'_'+predict_label+'.pk','wb') as f:
        pickle.dump(model,f)


def TreeModel(data,method,datatype,year,predict_label):
    valid_data = data.loc[data['year']==(year-1)]
    train_data = data.loc[data['year']<(year-1)]
    train_x = train_data.iloc[:,3:-2].values
    train_y = train_data.iloc[:,-2].values.reshape(-1,1)
    train_sample_weight = train_data.iloc[:,-1].values
    valid_x = valid_data.iloc[:,3:-2].values
    valid_y = valid_data.iloc[:,-2].values.reshape(-1,1)
    valid_sample_weight =  valid_data.iloc[:,-1].values
    
    error = np.inf
    if method=='RF':
        for n_estimators in range(10,100,10):
            for max_depth in range(3,7,1):
                for max_features in range(3,7,1):
                    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,random_state=1)
                    model.fit(train_x,train_y,sample_weight=train_sample_weight)
                    predict_y = model.predict_proba(valid_x)[:,1]
                    new_error = metrics.log_loss(valid_y,predict_y,sample_weight=valid_sample_weight)
                    if new_error < error:
                        error = new_error
                        select_model = copy.deepcopy(model)
  
        with open(r'models_final\RF'+'_'+datatype+'_'+str(year)+'_'+predict_label+'.pk','wb') as f:
            pickle.dump(select_model,f)

    elif method=='XGBoost':
        for n_estimators in range(10,100,20):
            for max_depth in range(3,7,1):
                for reg_lambda in [1e-3,1e-2,1e-1,1,10]:
                    model = xgb.XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,reg_lambda=reg_lambda,random_state=1)
                    model.fit(train_x,train_y,sample_weight=train_sample_weight)
                    predict_y = model.predict_proba(valid_x)[:,1]
                    new_error = metrics.log_loss(valid_y,predict_y,sample_weight=valid_sample_weight)
                    if new_error < error:
                        error = new_error
                        select_model = copy.deepcopy(model)
        with open(r'models_final\XGBoost'+'_'+datatype+'_'+str(year)+'_'+predict_label+'.pk','wb') as f:
            pickle.dump(select_model,f)


def DFFN(train_data1,neurons1,train_data2,neurons2,datatype,year,predict_label,ensemble=10):
    valid_data1 = train_data1.loc[train_data1['year']==(year-1)]
    train_data1 = train_data1.loc[train_data1['year']<(year-1)]
    train_x1 = torch.FloatTensor(train_data1.iloc[:,3:-2].values)
    valid_data2 = train_data2.loc[train_data2['year']==(year-1)]
    train_data2 = train_data2.loc[train_data2['year']<(year-1)]
    train_x2 = torch.FloatTensor(train_data2.iloc[:,3:-2].values)
    train_y = torch.FloatTensor(train_data1.iloc[:,-2].values.reshape(-1,1))
    train_sample_weight = torch.FloatTensor(train_data1.iloc[:,-1].values.reshape(-1,1))
    valid_x1 = torch.FloatTensor(valid_data1.iloc[:,3:-2].values)
    valid_x2 = torch.FloatTensor(valid_data2.iloc[:,3:-2].values)
    valid_y = torch.FloatTensor(valid_data1.iloc[:,-2].values.reshape(-1,1))
    valid_sample_weight =  torch.FloatTensor(valid_data1.iloc[:,-1].values.reshape(-1,1))
    for i in range(ensemble):

        setup_seed(i)
        dataset = Data.TensorDataset(train_x1,train_x2,train_y,train_sample_weight)
        train_loader = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=True) 
        
        Net = NN2(train_x1.shape[1],train_x2.shape[1],1,neurons1,neurons2)
        optimizer = torch.optim.Adam(Net.parameters())
        
        select_model = None
        error = np.inf
        j = 0
        for k in range(1000):       
            for step,(batch_x1,batch_x2,batch_y,sample_weight) in enumerate(train_loader): 
                if batch_x1.shape[0] < 64:
                    pass
                else:
                    prediction = Net(batch_x1,batch_x2)
                    loss_func = torch.nn.BCELoss(weight=sample_weight)
                    loss = loss_func(prediction,batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            j += 1
            Net.eval()
            predict_y = Net(valid_x1,valid_x2)
            loss_func = torch.nn.BCELoss(weight=valid_sample_weight)
            new_error = loss_func(predict_y,valid_y).data.numpy()
            Net.train()
            # print(new_error)
            if new_error < error:
                error = new_error
                select_model = copy.deepcopy(Net)
                j = 0
            if j> 50:
                break
        
        select_model.eval()
        with open(r'models_final\DFFN'+str(neurons1)+str(neurons2)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','wb') as f:
            pickle.dump(select_model,f)


def DFFN2(train_data1,neurons1,train_data2,neurons2,neurons,datatype,year,predict_label,ensemble=10):
    valid_data1 = train_data1.loc[train_data1['year']==(year-1)]
    train_data1 = train_data1.loc[train_data1['year']<(year-1)]
    train_x1 = torch.FloatTensor(train_data1.iloc[:,3:-2].values)
    valid_data2 = train_data2.loc[train_data2['year']==(year-1)]
    train_data2 = train_data2.loc[train_data2['year']<(year-1)]
    train_x2 = torch.FloatTensor(train_data2.iloc[:,3:-2].values)
    train_y = torch.FloatTensor(train_data1.iloc[:,-2].values.reshape(-1,1))
    train_sample_weight = torch.FloatTensor(train_data1.iloc[:,-1].values.reshape(-1,1))
    valid_x1 = torch.FloatTensor(valid_data1.iloc[:,3:-2].values)
    valid_x2 = torch.FloatTensor(valid_data2.iloc[:,3:-2].values)
    valid_y = torch.FloatTensor(valid_data1.iloc[:,-2].values.reshape(-1,1))
    valid_sample_weight =  torch.FloatTensor(valid_data1.iloc[:,-1].values.reshape(-1,1))
    for i in range(ensemble):
        setup_seed(i)
        dataset = Data.TensorDataset(train_x1,train_x2,train_y,train_sample_weight)
        train_loader = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=True) 
        
        Net = NN22(train_x1.shape[1],train_x2.shape[1],1,neurons1,neurons2,neurons)
        optimizer = torch.optim.Adam(Net.parameters())
        
        select_model = None
        error = np.inf
        j = 0
        for k in range(1000):       
            for step,(batch_x1,batch_x2,batch_y,sample_weight) in enumerate(train_loader): 
                if batch_x1.shape[0] < 64:
                    pass
                else:
                    prediction = Net(batch_x1,batch_x2)
                    loss_func = torch.nn.BCELoss(weight=sample_weight)
                    loss = loss_func(prediction,batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            j += 1
            Net.eval()
            predict_y = Net(valid_x1,valid_x2)
            loss_func = torch.nn.BCELoss(weight=valid_sample_weight)
            new_error = loss_func(predict_y,valid_y).data.numpy()
            Net.train()
            # print(new_error)
            if new_error < error:
                error = new_error
                select_model = copy.deepcopy(Net)
                j = 0
            if j> 50:
                break
        
        select_model.eval()
        with open(r'models_final\DFFN'+str(neurons1)+str(neurons2)+str(neurons)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','wb') as f:
            pickle.dump(select_model,f)


def TFFN(train_data1,neurons1,train_data2,neurons2,train_data3,neurons3,datatype,year,predict_label,ensemble=10):
    valid_data1 = train_data1.loc[train_data1['year']==(year-1)]
    train_data1 = train_data1.loc[train_data1['year']<(year-1)]
    train_x1 = torch.FloatTensor(train_data1.iloc[:,3:-2].values)
    valid_data2 = train_data2.loc[train_data2['year']==(year-1)]
    train_data2 = train_data2.loc[train_data2['year']<(year-1)]
    train_x2 = torch.FloatTensor(train_data2.iloc[:,3:-2].values)
    valid_data3 = train_data3.loc[train_data3['year']==(year-1)]
    train_data3 = train_data3.loc[train_data3['year']<(year-1)]
    train_x3 = torch.FloatTensor(train_data3.iloc[:,3:-2].values)
    train_y = torch.FloatTensor(train_data1.iloc[:,-2].values.reshape(-1,1))
    train_sample_weight = torch.FloatTensor(train_data1.iloc[:,-1].values.reshape(-1,1))
    valid_x1 = torch.FloatTensor(valid_data1.iloc[:,3:-2].values)
    valid_x2 = torch.FloatTensor(valid_data2.iloc[:,3:-2].values)
    valid_x3 = torch.FloatTensor(valid_data3.iloc[:,3:-2].values)
    valid_y = torch.FloatTensor(valid_data1.iloc[:,-2].values.reshape(-1,1))
    valid_sample_weight =  torch.FloatTensor(valid_data1.iloc[:,-1].values.reshape(-1,1))
    for i in range(ensemble):
        setup_seed(i)
        dataset = Data.TensorDataset(train_x1,train_x2,train_x3,train_y,train_sample_weight)
        train_loader = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=True) 
        
        Net = NN3(train_x1.shape[1],train_x2.shape[1],train_x3.shape[1],1,neurons1,neurons2,neurons3)
        optimizer = torch.optim.Adam(Net.parameters())
        
        select_model = None
        error = np.inf
        j = 0
        for k in range(1000):       
            for step,(batch_x1,batch_x2,batch_x3,batch_y,sample_weight) in enumerate(train_loader): 
                if batch_x1.shape[0] < 64:
                    pass
                else:
                    prediction = Net(batch_x1,batch_x2,batch_x3)
                    loss_func = torch.nn.BCELoss(weight=sample_weight)
                    loss = loss_func(prediction,batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            j += 1
            Net.eval()
            predict_y = Net(valid_x1,valid_x2,valid_x3)
            loss_func = torch.nn.BCELoss(weight=valid_sample_weight)
            new_error = loss_func(predict_y,valid_y).data.numpy()
            Net.train()
            # print(new_error)
            if new_error < error:
                error = new_error
                select_model = copy.deepcopy(Net)
                j = 0
            if j> 50:
                break
  
        select_model.eval()
        with open(r'models_final\TFFN'+str(neurons1)+str(neurons2)+str(neurons3)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','wb') as f:
            pickle.dump(select_model,f)


def TFFN2(train_data1,neurons1,train_data2,neurons2,train_data3,neurons3,neurons,datatype,year,predict_label,ensemble=10):
    valid_data1 = train_data1.loc[train_data1['year']==(year-1)]
    train_data1 = train_data1.loc[train_data1['year']<(year-1)]
    train_x1 = torch.FloatTensor(train_data1.iloc[:,3:-2].values)
    valid_data2 = train_data2.loc[train_data2['year']==(year-1)]
    train_data2 = train_data2.loc[train_data2['year']<(year-1)]
    train_x2 = torch.FloatTensor(train_data2.iloc[:,3:-2].values)
    valid_data3 = train_data3.loc[train_data3['year']==(year-1)]
    train_data3 = train_data3.loc[train_data3['year']<(year-1)]
    train_x3 = torch.FloatTensor(train_data3.iloc[:,3:-2].values)
    train_y = torch.FloatTensor(train_data1.iloc[:,-2].values.reshape(-1,1))
    train_sample_weight = torch.FloatTensor(train_data1.iloc[:,-1].values.reshape(-1,1))
    valid_x1 = torch.FloatTensor(valid_data1.iloc[:,3:-2].values)
    valid_x2 = torch.FloatTensor(valid_data2.iloc[:,3:-2].values)
    valid_x3 = torch.FloatTensor(valid_data3.iloc[:,3:-2].values)
    valid_y = torch.FloatTensor(valid_data1.iloc[:,-2].values.reshape(-1,1))
    valid_sample_weight =  torch.FloatTensor(valid_data1.iloc[:,-1].values.reshape(-1,1))

    for i in range(ensemble):
        setup_seed(i)
        dataset = Data.TensorDataset(train_x1,train_x2,train_x3,train_y,train_sample_weight)
        train_loader = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=True) 
        
        Net = NN32(train_x1.shape[1],train_x2.shape[1],train_x3.shape[1],1,neurons1,neurons2,neurons3,neurons)
        optimizer = torch.optim.Adam(Net.parameters())
        
        select_model = None
        error = np.inf
        j = 0
        for k in range(1000):       
            for step,(batch_x1,batch_x2,batch_x3,batch_y,sample_weight) in enumerate(train_loader): 
                if batch_x1.shape[0] < 64:
                    pass
                else:
                    prediction = Net(batch_x1,batch_x2,batch_x3)
                    loss_func = torch.nn.BCELoss(weight=sample_weight)
                    loss = loss_func(prediction,batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            j += 1
            Net.eval()
            predict_y = Net(valid_x1,valid_x2,valid_x3)
            loss_func = torch.nn.BCELoss(weight=valid_sample_weight)
            new_error = loss_func(predict_y,valid_y).data.numpy()
            Net.train()
            # print(new_error)
            if new_error < error:
                error = new_error
                select_model = copy.deepcopy(Net)
                j = 0
            if j> 50:
                break
  
        select_model.eval()
        with open(r'models_final\TFFN'+str(neurons1)+str(neurons2)+str(neurons3)+str(neurons)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','wb') as f:
            pickle.dump(select_model,f)


if __name__ == "__main__":  
    if 'models_final' not in os.listdir():
        os.mkdir('models_final')
#%% Table4
    #load datas
    datas = pd.read_csv(r'firm information.csv')
    macro_datas = pd.read_csv(r'macro.csv', index_col='Unnamed: 0')
    yield_datas = pd.read_csv(r'yield.csv', index_col='Unnamed: 0')
    #indicators
    factor_list = ['netpremiumswritten', 'grosspremiumswritten', 'totalassets','profitorlossbeforetax', 'capitalsurplus',
           'grosstechnicalreserves', 'totaldebtors', 'liquidassets','totalinvestments', 'nettechnicalreserves', 'totalliabilities',
           'profitorlossaftertax', 'numberoflinesofbusiness','numberofpremiumregions', 'numberofemployees', 'life', 'composite',
           'affiliated', 'mutual', 'totaldebtorstotalassets','liquidassetsnettechnicalreserves', 'liquidassetstotalliabilities',
           'totalinvestmenttotalliabilities','netpremiumswrittencapitalsurplus','nettechnicalreservescapitalsurpl',
           'grosspremiumswrittencapitalsurpl','grosstechnicalreservescapitalsur', 'totaldebtorscapitalsurplus',
           'leverageratio', 'rop', 'roa', 'roe', 'premiumretainratio','investmentasset', 'nettechnicalreservesgrosstechnic',
           'capitalsurplustotalassets', 'roebeforetax', 'roabeforetax','changeinasset', 'changeinnetpremium', 'changeingrosspremium',
           'movingaverageofroa', 'movingaverageofrop', 'movingaverageofroe','movingaverageofroabeforetax', 'movingaverageofroebeforetax',
           'movingstandarddeviationofroa', 'movingstandarddeviationofrop','movingstandarddeviationofroe', 'movingstandarddeviationofroabefo',
           'movingstandarddeviationofroebefo', 'riskedadjustedroa','riskedadjustedroe', 'riskedadjustedroabeforetax','riskedadjustedroebeforetax']
    ln_list = factor_list[:15]
    dummy_list = factor_list[15:19]
    ratio_list = factor_list[19:]
    macro_list = ['yearly return of MSCI', 'change in IMF one year T-bill rate', 'GDP', 'real GDP growth', 'GDP per capita', 'inflation',
           'unemployment rate', 'yearly wage (insurance sector)', 'change in population', 'change in current account',
           'change in yearly wage', 'change in unemployment rate', 'change in exchange rate', 'growth of export', 'growth of import',
           'growth of consumption per capita', 'growth of capital', 'growth of industry production', 'growth of tax revenue',
           'growth of gov expenditure', 'growth of carbon emission', 'growth of labor in agriculture industry',
           'growth of population with access to Internet', 'Labor force / population', 'Capital Ratio of Banks',
           'Domestic credit / GDP', 'Domestic private credit / GDP', 'insurance penetration', 'insurance density']
    yield_list = [str(i)+'Y' for i in range(1,11)]


    predict_label = 'failure'
    for datatype in ['firm', 'firm+yield', 'firm+macro', 'firm+macro+yield']:
        for year in range(2010,2013):
            t0 = time.time()
            print(year)
            data = datas.loc[datas['year']<=year]
            for factor in ln_list:
                if data[factor].min() > 1:
                    data[factor] = np.log(data[factor])
                else:
                    data[factor] = np.log(data[factor]-data[factor].min()+1)
    
            normal_list = [x for x in factor_list if x not in dummy_list]
            for factor in normal_list:
                data[factor] = np.clip(data[factor],np.quantile(data[factor].dropna(),0.01),np.quantile(data[factor].dropna(),0.99))
                data[factor] = (data[factor]-data[factor].mean())/data[factor].std()
                data[factor] = data[factor].fillna(0)
            # raise
            yield_data = yield_datas.loc[yield_datas['year']<=year]
            macro_data = macro_datas.loc[macro_datas['year']<=year]
            macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']] = np.log(macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']])
            macro_data[macro_list] = (macro_data[macro_list] - macro_data[macro_list].mean())/macro_data[macro_list].std()
            
            data = pd.merge(data,yield_data,on=['country','year'],how='left')
            data = pd.merge(data,macro_data,on=['country','year'],how='left')
                 
            
            if datatype == 'firm':
                data = data[['ambnumber','year','country']+factor_list+[predict_label]]
            if datatype == 'firm+yield':
                data = data[['ambnumber','year','country']+factor_list+yield_list+[predict_label]]
            if datatype == 'firm+macro':
                data = data[['ambnumber','year','country']+factor_list+macro_list+[predict_label]]   
            if datatype == 'firm+macro+yield':
                data = data[['ambnumber','year','country']+factor_list+yield_list+macro_list+[predict_label]]
            
            #cost-sensitive learning
            train_data = data.loc[data['year']<year]
            train_data['sample_weight'] = 1
            train_data.loc[train_data[predict_label]==1,'sample_weight'] = train_data.shape[0]/train_data[predict_label].sum() -1

            # training models
            logistic(train_data,datatype,year,predict_label)
            
            TreeModel(train_data, 'RF',datatype,year,predict_label)
            TreeModel(train_data, 'XGBoost',datatype,year,predict_label)
            
    
            FFN(train_data,[4],datatype,year,predict_label)
            FFN(train_data,[4,2],datatype,year,predict_label)
            FFN(train_data,[4,3,2],datatype,year,predict_label)
            FFN(train_data,[8,6,4,2],datatype,year,predict_label)
            FFN(train_data,[8,6,4,3,2],datatype,year,predict_label)
            
            
            if datatype == 'firm+yield':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                DFFN(train_data1,[4,3,2],train_data2,[4,3,2],datatype,year,predict_label)
                DFFN2(train_data1,[4,3,2],train_data2,[4,3,2],[3],datatype,year,predict_label)
            
            if datatype == 'firm+macro':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                DFFN(train_data1,[4,3,2],train_data2,[4,3,2],datatype,year,predict_label)
                DFFN2(train_data1,[4,3,2],train_data2,[4,3,2],[3],datatype,year,predict_label)
            
            if datatype == 'firm+macro+yield':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                train_data3 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                TFFN(train_data1,[4,3,2],train_data2,[4,3,2],train_data3,[4,3,2],datatype,year,predict_label)
                TFFN2(train_data1,[4,3,2],train_data2,[4,3,2],train_data3,[4,3,2],[3],datatype,year,predict_label)
       
   
#%% Table5
    #load datas
    datas = pd.read_csv(r'firm information.csv')
    macro_datas = pd.read_csv(r'macro.csv', index_col='Unnamed: 0')
    yield_datas = pd.read_csv(r'yield.csv', index_col='Unnamed: 0')
    #indicators
    factor_list = ['netpremiumswritten', 'grosspremiumswritten', 'totalassets','profitorlossbeforetax', 'capitalsurplus',
           'grosstechnicalreserves', 'totaldebtors', 'liquidassets','totalinvestments', 'nettechnicalreserves', 'totalliabilities',
           'profitorlossaftertax', 'numberoflinesofbusiness','numberofpremiumregions', 'numberofemployees', 'life', 'composite',
           'affiliated', 'mutual', 'totaldebtorstotalassets','liquidassetsnettechnicalreserves', 'liquidassetstotalliabilities',
           'totalinvestmenttotalliabilities','netpremiumswrittencapitalsurplus','nettechnicalreservescapitalsurpl',
           'grosspremiumswrittencapitalsurpl','grosstechnicalreservescapitalsur', 'totaldebtorscapitalsurplus',
           'leverageratio', 'rop', 'roa', 'roe', 'premiumretainratio','investmentasset', 'nettechnicalreservesgrosstechnic',
           'capitalsurplustotalassets', 'roebeforetax', 'roabeforetax','changeinasset', 'changeinnetpremium', 'changeingrosspremium',
           'movingaverageofroa', 'movingaverageofrop', 'movingaverageofroe','movingaverageofroabeforetax', 'movingaverageofroebeforetax',
           'movingstandarddeviationofroa', 'movingstandarddeviationofrop','movingstandarddeviationofroe', 'movingstandarddeviationofroabefo',
           'movingstandarddeviationofroebefo', 'riskedadjustedroa','riskedadjustedroe', 'riskedadjustedroabeforetax','riskedadjustedroebeforetax']
    ln_list = factor_list[:15]
    dummy_list = factor_list[15:19]
    ratio_list = factor_list[19:]
    macro_list = ['yearly return of MSCI', 'change in IMF one year T-bill rate', 'GDP', 'real GDP growth', 'GDP per capita', 'inflation',
           'unemployment rate', 'yearly wage (insurance sector)', 'change in population', 'change in current account',
           'change in yearly wage', 'change in unemployment rate', 'change in exchange rate', 'growth of export', 'growth of import',
           'growth of consumption per capita', 'growth of capital', 'growth of industry production', 'growth of tax revenue',
           'growth of gov expenditure', 'growth of carbon emission', 'growth of labor in agriculture industry',
           'growth of population with access to Internet', 'Labor force / population', 'Capital Ratio of Banks',
           'Domestic credit / GDP', 'Domestic private credit / GDP', 'insurance penetration', 'insurance density']
    yield_list = [str(i)+'Y' for i in range(1,11)]
    
    predict_label = 'distressed'
    for datatype in ['firm', 'firm+yield', 'firm+macro', 'firm+macro+yield']:
        for year in range(2010,2013):
            t0 = time.time()
            print(year)
            data = datas.loc[datas['year']<=year]
            for factor in ln_list:
                if data[factor].min() > 1:
                    data[factor] = np.log(data[factor])
                else:
                    data[factor] = np.log(data[factor]-data[factor].min()+1)
    
            normal_list = [x for x in factor_list if x not in dummy_list]
            for factor in normal_list:
                data[factor] = np.clip(data[factor],np.quantile(data[factor].dropna(),0.01),np.quantile(data[factor].dropna(),0.99))
                data[factor] = (data[factor]-data[factor].mean())/data[factor].std()
                data[factor] = data[factor].fillna(0)
            # raise
            yield_data = yield_datas.loc[yield_datas['year']<=year]
            macro_data = macro_datas.loc[macro_datas['year']<=year]
            macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']] = np.log(macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']])
            macro_data[macro_list] = (macro_data[macro_list] - macro_data[macro_list].mean())/macro_data[macro_list].std()
            
            data = pd.merge(data,yield_data,on=['country','year'],how='left')
            data = pd.merge(data,macro_data,on=['country','year'],how='left')
                 
            
            if datatype == 'firm':
                data = data[['ambnumber','year','country']+factor_list+[predict_label]]
            if datatype == 'firm+yield':
                data = data[['ambnumber','year','country']+factor_list+yield_list+[predict_label]]
            if datatype == 'firm+macro':
                data = data[['ambnumber','year','country']+factor_list+macro_list+[predict_label]]   
            if datatype == 'firm+macro+yield':
                data = data[['ambnumber','year','country']+factor_list+yield_list+macro_list+[predict_label]]
            
            #cost-sensitive learning
            train_data = data.loc[data['year']<year]
            train_data['sample_weight'] = 1
            train_data.loc[train_data[predict_label]==1,'sample_weight'] = train_data.shape[0]/train_data[predict_label].sum() -1

            # training models
            logistic(train_data,datatype,year,predict_label)
            
            TreeModel(train_data, 'RF',datatype,year,predict_label)
            TreeModel(train_data, 'XGBoost',datatype,year,predict_label)
            
    
            FFN(train_data,[4],datatype,year,predict_label)
            FFN(train_data,[4,2],datatype,year,predict_label)
            FFN(train_data,[4,3,2],datatype,year,predict_label)
            FFN(train_data,[8,6,4,2],datatype,year,predict_label)
            FFN(train_data,[8,6,4,3,2],datatype,year,predict_label)
            
            
            if datatype == 'firm+yield':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                DFFN(train_data1,[4,3,2],train_data2,[4,3,2],datatype,year,predict_label)
                DFFN2(train_data1,[4,3,2],train_data2,[4,3,2],[3],datatype,year,predict_label)
            
            if datatype == 'firm+macro':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                DFFN(train_data1,[4,3,2],train_data2,[4,3,2],datatype,year,predict_label)
                DFFN2(train_data1,[4,3,2],train_data2,[4,3,2],[3],datatype,year,predict_label)
            
            if datatype == 'firm+macro+yield':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                train_data3 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                TFFN(train_data1,[4,3,2],train_data2,[4,3,2],train_data3,[4,3,2],datatype,year,predict_label)
                TFFN2(train_data1,[4,3,2],train_data2,[4,3,2],train_data3,[4,3,2],[3],datatype,year,predict_label)
       

#%% Table6
    #load datas
    datas = pd.read_csv(r'firm information_life.csv')
    datas = datas.rename(columns={'failure':'failure_life'})
    macro_datas = pd.read_csv(r'macro.csv', index_col='Unnamed: 0')
    yield_datas = pd.read_csv(r'yield.csv', index_col='Unnamed: 0')
    #indicators
    factor_list = ['netpremiumswritten', 'grosspremiumswritten', 'totalassets', 'profitorlossbeforetax', 'capitalsurplus',
           'grosstechnicalreserves', 'totaldebtors', 'liquidassets', 'totalinvestments', 'nettechnicalreserves', 'totalliabilities',
           'profitorlossaftertax', 'benefitspaid', 'netexpenseslife', 'numberoflinesofbusiness', 'numberofpremiumregions',
           'numberofemployees', 'affiliated', 'mutual','totaldebtorstotalassets', 'liquidassetsnettechnicalreserves',
           'liquidassetstotalliabilities', 'totalinvestmenttotalliabilities', 'netpremiumswrittencapitalsurplus',
           'nettechnicalreservescapitalsurpl', 'grosspremiumswrittencapitalsurpl', 'grosstechnicalreservescapitalsur', 'totaldebtorscapitalsurplus',
           'leverageratio', 'rop', 'roa', 'roe', 'premiumretainratio', 'investmentasset', 'nettechnicalreservesgrosstechnic',
           'capitalsurplustotalassets', 'roebeforetax', 'roabeforetax', 'changeinasset', 'changeinnetpremium', 'changeingrosspremium',
           'movingaverageofroa', 'movingaverageofrop', 'movingaverageofroe', 'movingaverageofroabeforetax', 'movingaverageofroebeforetax',
           'movingstandarddeviationofroa', 'movingstandarddeviationofrop', 'movingstandarddeviationofroe', 'movingstandarddeviationofroabefo',
           'movingstandarddeviationofroebefo', 'riskedadjustedroa', 'riskedadjustedroe', 'riskedadjustedroabeforetax',
           'riskedadjustedroebeforetax', 'benefitspaidnetpremiumswritten','expenseratio', 'combinedratiolife']
    ln_list = factor_list[:17]
    dummy_list = factor_list[17:19]
    ratio_list = factor_list[19:]
    macro_list = ['yearly return of MSCI', 'change in IMF one year T-bill rate', 'GDP', 'real GDP growth', 'GDP per capita', 'inflation',
           'unemployment rate', 'yearly wage (insurance sector)', 'change in population', 'change in current account',
           'change in yearly wage', 'change in unemployment rate', 'change in exchange rate', 'growth of export', 'growth of import',
           'growth of consumption per capita', 'growth of capital', 'growth of industry production', 'growth of tax revenue',
           'growth of gov expenditure', 'growth of carbon emission', 'growth of labor in agriculture industry',
           'growth of population with access to Internet', 'Labor force / population', 'Capital Ratio of Banks',
           'Domestic credit / GDP', 'Domestic private credit / GDP', 'insurance penetration', 'insurance density']
    yield_list = [str(i)+'Y' for i in range(1,11)]
    
    predict_label = 'failure_life'
    for datatype in ['firm', 'firm+yield', 'firm+macro', 'firm+macro+yield']:
        for year in range(2010,2013):
            t0 = time.time()
            print(year)
            data = datas.loc[datas['year']<=year]
            for factor in ln_list:
                if data[factor].min() > 1:
                    data[factor] = np.log(data[factor])
                else:
                    data[factor] = np.log(data[factor]-data[factor].min()+1)
    
            normal_list = [x for x in factor_list if x not in dummy_list]
            for factor in normal_list:
                data[factor] = np.clip(data[factor],np.quantile(data[factor].dropna(),0.01),np.quantile(data[factor].dropna(),0.99))
                data[factor] = (data[factor]-data[factor].mean())/data[factor].std()
                data[factor] = data[factor].fillna(0)
            # raise
            yield_data = yield_datas.loc[yield_datas['year']<=year]
            macro_data = macro_datas.loc[macro_datas['year']<=year]
            macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']] = np.log(macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']])
            macro_data[macro_list] = (macro_data[macro_list] - macro_data[macro_list].mean())/macro_data[macro_list].std()
            
            data = pd.merge(data,yield_data,on=['country','year'],how='left')
            data = pd.merge(data,macro_data,on=['country','year'],how='left')
                 
            
            if datatype == 'firm':
                data = data[['ambnumber','year','country']+factor_list+[predict_label]]
            if datatype == 'firm+yield':
                data = data[['ambnumber','year','country']+factor_list+yield_list+[predict_label]]
            if datatype == 'firm+macro':
                data = data[['ambnumber','year','country']+factor_list+macro_list+[predict_label]]   
            if datatype == 'firm+macro+yield':
                data = data[['ambnumber','year','country']+factor_list+yield_list+macro_list+[predict_label]]
            
            #cost-sensitive learning
            train_data = data.loc[data['year']<year]
            train_data['sample_weight'] = 1
            train_data.loc[train_data[predict_label]==1,'sample_weight'] = train_data.shape[0]/train_data[predict_label].sum() -1

            # training models
            logistic(train_data,datatype,year,predict_label)
            
            TreeModel(train_data, 'RF',datatype,year,predict_label)
            TreeModel(train_data, 'XGBoost',datatype,year,predict_label)
            
    
            FFN(train_data,[4],datatype,year,predict_label)
            FFN(train_data,[4,2],datatype,year,predict_label)
            FFN(train_data,[4,3,2],datatype,year,predict_label)
            FFN(train_data,[8,6,4,2],datatype,year,predict_label)
            FFN(train_data,[8,6,4,3,2],datatype,year,predict_label)
            
            
            if datatype == 'firm+yield':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                DFFN(train_data1,[4,3,2],train_data2,[4,3,2],datatype,year,predict_label)
                DFFN2(train_data1,[4,3,2],train_data2,[4,3,2],[3],datatype,year,predict_label)
            
            if datatype == 'firm+macro':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                DFFN(train_data1,[4,3,2],train_data2,[4,3,2],datatype,year,predict_label)
                DFFN2(train_data1,[4,3,2],train_data2,[4,3,2],[3],datatype,year,predict_label)
            
            if datatype == 'firm+macro+yield':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                train_data3 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                TFFN(train_data1,[4,3,2],train_data2,[4,3,2],train_data3,[4,3,2],datatype,year,predict_label)
                TFFN2(train_data1,[4,3,2],train_data2,[4,3,2],train_data3,[4,3,2],[3],datatype,year,predict_label)
       

#%% Table7
    #load datas
    datas = pd.read_csv(r'firm information_nonlife.csv')
    datas = datas.rename(columns={'failure':'failure_nonlife'})
    macro_datas = pd.read_csv(r'macro.csv', index_col='Unnamed: 0')
    yield_datas = pd.read_csv(r'yield.csv', index_col='Unnamed: 0')
    #indicators
    factor_list = ['netpremiumswritten', 'grosspremiumswritten', 'totalassets', 'profitorlossbeforetax', 'capitalsurplus',
           'grosstechnicalreserves', 'totaldebtors', 'liquidassets', 'totalinvestments', 'nettechnicalreserves', 'totalliabilities',
           'profitorlossaftertax', 'netexpenses', 'numberoflinesofbusiness', 'numberofpremiumregions', 'numberofemployees', 'affiliated',
           'mutual', 'totaldebtorstotalassets', 'liquidassetsnettechnicalreserves', 'liquidassetstotalliabilities', 'totalinvestmenttotalliabilities',
           'netpremiumswrittencapitalsurplus', 'nettechnicalreservescapitalsurpl', 'grosspremiumswrittencapitalsurpl',
           'grosstechnicalreservescapitalsur', 'totaldebtorscapitalsurplus', 'leverageratio', 'rop', 'roa', 'roe', 'premiumretainratio',
           'investmentasset', 'nettechnicalreservesgrosstechnic', 'capitalsurplustotalassets', 'roebeforetax', 'roabeforetax',
           'changeinasset', 'changeinnetpremium', 'changeingrosspremium', 'movingaverageofroa', 'movingaverageofrop', 'movingaverageofroe',
           'movingaverageofroabeforetax', 'movingaverageofroebeforetax', 'movingstandarddeviationofroa', 'movingstandarddeviationofrop',
           'movingstandarddeviationofroe', 'movingstandarddeviationofroabefo', 'movingstandarddeviationofroebefo', 'riskedadjustedroa',
           'riskedadjustedroe', 'riskedadjustedroabeforetax', 'riskedadjustedroebeforetax', 'lossratio', 'operatingexpenseratio',
           'combinedratio', 'netinvestmentincome', 'operatingratio']
    ln_list = factor_list[:16]
    dummy_list = factor_list[16:18]
    ratio_list = factor_list[18:]
    macro_list = ['yearly return of MSCI', 'change in IMF one year T-bill rate', 'GDP', 'real GDP growth', 'GDP per capita', 'inflation',
           'unemployment rate', 'yearly wage (insurance sector)', 'change in population', 'change in current account',
           'change in yearly wage', 'change in unemployment rate', 'change in exchange rate', 'growth of export', 'growth of import',
           'growth of consumption per capita', 'growth of capital', 'growth of industry production', 'growth of tax revenue',
           'growth of gov expenditure', 'growth of carbon emission', 'growth of labor in agriculture industry',
           'growth of population with access to Internet', 'Labor force / population', 'Capital Ratio of Banks',
           'Domestic credit / GDP', 'Domestic private credit / GDP', 'insurance penetration', 'insurance density']
    yield_list = [str(i)+'Y' for i in range(1,11)]
    
    predict_label = 'failure_nonlife'
    for datatype in ['firm', 'firm+yield', 'firm+macro', 'firm+macro+yield']:
        for year in range(2010,2013):
            t0 = time.time()
            print(year)
            data = datas.loc[datas['year']<=year]
            for factor in ln_list:
                if data[factor].min() > 1:
                    data[factor] = np.log(data[factor])
                else:
                    data[factor] = np.log(data[factor]-data[factor].min()+1)
    
            normal_list = [x for x in factor_list if x not in dummy_list]
            for factor in normal_list:
                data[factor] = np.clip(data[factor],np.quantile(data[factor].dropna(),0.01),np.quantile(data[factor].dropna(),0.99))
                data[factor] = (data[factor]-data[factor].mean())/data[factor].std()
                data[factor] = data[factor].fillna(0)
            # raise
            yield_data = yield_datas.loc[yield_datas['year']<=year]
            macro_data = macro_datas.loc[macro_datas['year']<=year]
            macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']] = np.log(macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']])
            macro_data[macro_list] = (macro_data[macro_list] - macro_data[macro_list].mean())/macro_data[macro_list].std()
            
            data = pd.merge(data,yield_data,on=['country','year'],how='left')
            data = pd.merge(data,macro_data,on=['country','year'],how='left')
                 
            
            if datatype == 'firm':
                data = data[['ambnumber','year','country']+factor_list+[predict_label]]
            if datatype == 'firm+yield':
                data = data[['ambnumber','year','country']+factor_list+yield_list+[predict_label]]
            if datatype == 'firm+macro':
                data = data[['ambnumber','year','country']+factor_list+macro_list+[predict_label]]   
            if datatype == 'firm+macro+yield':
                data = data[['ambnumber','year','country']+factor_list+yield_list+macro_list+[predict_label]]
            
            #cost-sensitive learning
            train_data = data.loc[data['year']<year]
            train_data['sample_weight'] = 1
            train_data.loc[train_data[predict_label]==1,'sample_weight'] = train_data.shape[0]/train_data[predict_label].sum() -1

            # training models
            logistic(train_data,datatype,year,predict_label)
            
            TreeModel(train_data, 'RF',datatype,year,predict_label)
            TreeModel(train_data, 'XGBoost',datatype,year,predict_label)
            
    
            FFN(train_data,[4],datatype,year,predict_label)
            FFN(train_data,[4,2],datatype,year,predict_label)
            FFN(train_data,[4,3,2],datatype,year,predict_label)
            FFN(train_data,[8,6,4,2],datatype,year,predict_label)
            FFN(train_data,[8,6,4,3,2],datatype,year,predict_label)
            
            
            if datatype == 'firm+yield':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                DFFN(train_data1,[4,3,2],train_data2,[4,3,2],datatype,year,predict_label)
                DFFN2(train_data1,[4,3,2],train_data2,[4,3,2],[3],datatype,year,predict_label)
            
            if datatype == 'firm+macro':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                DFFN(train_data1,[4,3,2],train_data2,[4,3,2],datatype,year,predict_label)
                DFFN2(train_data1,[4,3,2],train_data2,[4,3,2],[3],datatype,year,predict_label)
            
            if datatype == 'firm+macro+yield':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                train_data3 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                TFFN(train_data1,[4,3,2],train_data2,[4,3,2],train_data3,[4,3,2],datatype,year,predict_label)
                TFFN2(train_data1,[4,3,2],train_data2,[4,3,2],train_data3,[4,3,2],[3],datatype,year,predict_label)
       

#%% Table8
    #load datas
    datas = pd.read_csv(r'firm information_dropmergers.csv')
    datas = datas.rename(columns={'failure':'failure_dropmergers'})
    macro_datas = pd.read_csv(r'macro.csv', index_col='Unnamed: 0')
    yield_datas = pd.read_csv(r'yield.csv', index_col='Unnamed: 0')
    #indicators
    factor_list = ['netpremiumswritten', 'grosspremiumswritten', 'totalassets','profitorlossbeforetax', 'capitalsurplus',
           'grosstechnicalreserves', 'totaldebtors', 'liquidassets','totalinvestments', 'nettechnicalreserves', 'totalliabilities',
           'profitorlossaftertax', 'numberoflinesofbusiness','numberofpremiumregions', 'numberofemployees', 'life', 'composite',
           'affiliated', 'mutual', 'totaldebtorstotalassets','liquidassetsnettechnicalreserves', 'liquidassetstotalliabilities',
           'totalinvestmenttotalliabilities','netpremiumswrittencapitalsurplus','nettechnicalreservescapitalsurpl',
           'grosspremiumswrittencapitalsurpl','grosstechnicalreservescapitalsur', 'totaldebtorscapitalsurplus',
           'leverageratio', 'rop', 'roa', 'roe', 'premiumretainratio','investmentasset', 'nettechnicalreservesgrosstechnic',
           'capitalsurplustotalassets', 'roebeforetax', 'roabeforetax','changeinasset', 'changeinnetpremium', 'changeingrosspremium',
           'movingaverageofroa', 'movingaverageofrop', 'movingaverageofroe','movingaverageofroabeforetax', 'movingaverageofroebeforetax',
           'movingstandarddeviationofroa', 'movingstandarddeviationofrop','movingstandarddeviationofroe', 'movingstandarddeviationofroabefo',
           'movingstandarddeviationofroebefo', 'riskedadjustedroa','riskedadjustedroe', 'riskedadjustedroabeforetax','riskedadjustedroebeforetax']
    ln_list = factor_list[:15]
    dummy_list = factor_list[15:19]
    ratio_list = factor_list[19:]
    macro_list = ['yearly return of MSCI', 'change in IMF one year T-bill rate', 'GDP', 'real GDP growth', 'GDP per capita', 'inflation',
           'unemployment rate', 'yearly wage (insurance sector)', 'change in population', 'change in current account',
           'change in yearly wage', 'change in unemployment rate', 'change in exchange rate', 'growth of export', 'growth of import',
           'growth of consumption per capita', 'growth of capital', 'growth of industry production', 'growth of tax revenue',
           'growth of gov expenditure', 'growth of carbon emission', 'growth of labor in agriculture industry',
           'growth of population with access to Internet', 'Labor force / population', 'Capital Ratio of Banks',
           'Domestic credit / GDP', 'Domestic private credit / GDP', 'insurance penetration', 'insurance density']
    yield_list = [str(i)+'Y' for i in range(1,11)]
    
    predict_label = 'failure_dropmergers'
    for datatype in ['firm', 'firm+yield', 'firm+macro', 'firm+macro+yield']:
        for year in range(2010,2013):
            t0 = time.time()
            print(year)
            data = datas.loc[datas['year']<=year]
            for factor in ln_list:
                if data[factor].min() > 1:
                    data[factor] = np.log(data[factor])
                else:
                    data[factor] = np.log(data[factor]-data[factor].min()+1)
    
            normal_list = [x for x in factor_list if x not in dummy_list]
            for factor in normal_list:
                data[factor] = np.clip(data[factor],np.quantile(data[factor].dropna(),0.01),np.quantile(data[factor].dropna(),0.99))
                data[factor] = (data[factor]-data[factor].mean())/data[factor].std()
                data[factor] = data[factor].fillna(0)
            # raise
            yield_data = yield_datas.loc[yield_datas['year']<=year]
            macro_data = macro_datas.loc[macro_datas['year']<=year]
            macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']] = np.log(macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']])
            macro_data[macro_list] = (macro_data[macro_list] - macro_data[macro_list].mean())/macro_data[macro_list].std()
            
            data = pd.merge(data,yield_data,on=['country','year'],how='left')
            data = pd.merge(data,macro_data,on=['country','year'],how='left')
                 
            
            if datatype == 'firm':
                data = data[['ambnumber','year','country']+factor_list+[predict_label]]
            if datatype == 'firm+yield':
                data = data[['ambnumber','year','country']+factor_list+yield_list+[predict_label]]
            if datatype == 'firm+macro':
                data = data[['ambnumber','year','country']+factor_list+macro_list+[predict_label]]   
            if datatype == 'firm+macro+yield':
                data = data[['ambnumber','year','country']+factor_list+yield_list+macro_list+[predict_label]]
            
            #cost-sensitive learning
            train_data = data.loc[data['year']<year]
            train_data['sample_weight'] = 1
            train_data.loc[train_data[predict_label]==1,'sample_weight'] = train_data.shape[0]/train_data[predict_label].sum() -1

            # training models
            logistic(train_data,datatype,year,predict_label)
            
            TreeModel(train_data, 'RF',datatype,year,predict_label)
            TreeModel(train_data, 'XGBoost',datatype,year,predict_label)
            
    
            FFN(train_data,[4],datatype,year,predict_label)
            FFN(train_data,[4,2],datatype,year,predict_label)
            FFN(train_data,[4,3,2],datatype,year,predict_label)
            FFN(train_data,[8,6,4,2],datatype,year,predict_label)
            FFN(train_data,[8,6,4,3,2],datatype,year,predict_label)
            
            
            if datatype == 'firm+yield':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                DFFN(train_data1,[4,3,2],train_data2,[4,3,2],datatype,year,predict_label)
                DFFN2(train_data1,[4,3,2],train_data2,[4,3,2],[3],datatype,year,predict_label)
            
            if datatype == 'firm+macro':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                DFFN(train_data1,[4,3,2],train_data2,[4,3,2],datatype,year,predict_label)
                DFFN2(train_data1,[4,3,2],train_data2,[4,3,2],[3],datatype,year,predict_label)
            
            if datatype == 'firm+macro+yield':
                train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                train_data3 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                TFFN(train_data1,[4,3,2],train_data2,[4,3,2],train_data3,[4,3,2],datatype,year,predict_label)
                TFFN2(train_data1,[4,3,2],train_data2,[4,3,2],train_data3,[4,3,2],[3],datatype,year,predict_label)
       