# -*- coding: utf-8 -*-


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
import os 
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

with pd.ExcelWriter('tables and figures/TableA.1_10AUC.xlsx') as writer:
    for k in range(10):
        
        def FFN(test_data,neurons,datatype,year,predict_label,ensemble=10):
            X_test = torch.FloatTensor(test_data.iloc[:,3:-1].values)
            count=0
            for i in range(ensemble):
                if i == k:  # 跳过编号为k的文件
                    continue       
                if count >= 9:  # 只抽取9个文件
                    break       
                with open(r'models_final\FFN'+str(neurons)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','rb') as f:
                    select_model = pickle.load(f)
                if count==0:
                    result = select_model(X_test).data.numpy()
                else:
                    result += select_model(X_test).data.numpy()
                count+=1
            return result/9
        
        def DFFN(test_data1,neurons1,test_data2,neurons2,datatype,year,predict_label,ensemble=10):
            X_test1 = torch.FloatTensor(test_data1.iloc[:,3:-1].values)
            X_test2 = torch.FloatTensor(test_data2.iloc[:,3:-1].values)
            count=0
            for i in range(ensemble):
                if i == k:  # 跳过编号为k的文件
                    continue       
                if count >= 9:  # 只抽取9个文件
                    break      
                with open(r'models_final\DFFN'+str(neurons1)+str(neurons2)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','rb') as f:
                    select_model = pickle.load(f)
                    
                if count==0:
                    result = select_model(X_test1,X_test2).data.numpy()
                else:
                    result += select_model(X_test1,X_test2).data.numpy()
                count+=1
            return result/9
        
        def DFFN2(test_data1,neurons1,test_data2,neurons2,neurons,datatype,year,predict_label,ensemble=10):
            X_test1 = torch.FloatTensor(test_data1.iloc[:,3:-1].values)
            X_test2 = torch.FloatTensor(test_data2.iloc[:,3:-1].values)
            count=0
            for i in range(ensemble):
                if i == k:  # 跳过编号为k的文件
                    continue       
                if count >= 9:  # 只抽取9个文件
                    break      
                with open(r'models_final\DFFN'+str(neurons1)+str(neurons2)+str(neurons)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','rb') as f:
                    select_model = pickle.load(f)
                if count==0:
                    result = select_model(X_test1,X_test2).data.numpy()
                else:
                    result += select_model(X_test1,X_test2).data.numpy()
                count+=1
            return result/9
        
        def TFFN(test_data1,neurons1,test_data2,neurons2,test_data3,neurons3,datatype,year,predict_label,ensemble=10):
            X_test1 = torch.FloatTensor(test_data1.iloc[:,3:-1].values)
            X_test2 = torch.FloatTensor(test_data2.iloc[:,3:-1].values)
            X_test3 = torch.FloatTensor(test_data3.iloc[:,3:-1].values)
            count=0
            for i in range(ensemble):
                if i == k:  # 跳过编号为k的文件
                    continue       
                if count >= 9:  # 只抽取9个文件
                    break      
                with open(r'models_final\TFFN'+str(neurons1)+str(neurons2)+str(neurons3)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','rb') as f:
                    select_model = pickle.load(f)
        
                if count==0:
                    result = select_model(X_test1,X_test2,X_test3).data.numpy()
                else:
                    result += select_model(X_test1,X_test2,X_test3).data.numpy()
                count+=1
            return result/9
        
        def TFFN2(test_data1,neurons1,test_data2,neurons2,test_data3,neurons3,neurons,datatype,year,predict_label,ensemble=10):
            X_test1 = torch.FloatTensor(test_data1.iloc[:,3:-1].values)
            X_test2 = torch.FloatTensor(test_data2.iloc[:,3:-1].values)
            X_test3 = torch.FloatTensor(test_data3.iloc[:,3:-1].values)
            count=0
            for i in range(ensemble):
                if i == k:  # 跳过编号为k的文件
                    continue       
                if count >= 9:  # 只抽取9个文件
                    break      
                with open(r'models_final\TFFN'+str(neurons1)+str(neurons2)+str(neurons3)+str(neurons)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','rb') as f:
                    select_model = pickle.load(f)
                         
                if count==0:
                    result = select_model(X_test1,X_test2,X_test3).data.numpy()
                else:
                    result += select_model(X_test1,X_test2,X_test3).data.numpy()
                count+=1
            return result/9
        
        if __name__ == "__main__":  
            ####### Table #######
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
            
            Table = None
            predict_label = 'failure'
            for datatype in ['firm', 'firm+yield', 'firm+macro', 'firm+macro+yield']:
                results = None
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
                    test_data = data.loc[data['year']==year]
                    result = test_data[['ambnumber','year',predict_label]]
                    # raise
                    result['NN[4]'] = FFN(test_data,[4],datatype,year,predict_label)
                    result['NN[4,2]'] = FFN(test_data,[4,2],datatype,year,predict_label)
                    result['NN[4,3,2]'] = FFN(test_data,[4,3,2],datatype,year,predict_label)
                    result['NN[8,6,4,2]'] = FFN(test_data,[8,6,4,2],datatype,year,predict_label)
                    result['NN[8,6,4,3,2]'] = FFN(test_data,[8,6,4,3,2],datatype,year,predict_label)
                    
                    if datatype == 'firm':
                        result['SANN0'] = result['NN[4,3,2]']
                        result['SANN1'] = result['NN[4,3,2]']
                    
                    if datatype == 'firm+yield':
                        train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                        train_data2 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                        test_data1 = test_data[['ambnumber','year','country']+factor_list+[predict_label]]
                        test_data2 = test_data[['ambnumber','year','country']+yield_list+[predict_label]]
                        result['SANN0'] = DFFN(test_data1,[4,3,2],test_data2,[4,3,2],datatype,year,predict_label)
                        result['SANN1'] = DFFN2(test_data1,[4,3,2],test_data2,[4,3,2],[3],datatype,year,predict_label)
                    
                    if datatype == 'firm+macro':
                        train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                        train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                        test_data1 = test_data[['ambnumber','year','country']+factor_list+[predict_label]]
                        test_data2 = test_data[['ambnumber','year','country']+macro_list+[predict_label]]
                        result['SANN0'] = DFFN(test_data1,[4,3,2],test_data2,[4,3,2],datatype,year,predict_label)
                        result['SANN1'] = DFFN2(test_data1,[4,3,2],test_data2,[4,3,2],[3],datatype,year,predict_label)
                    
                    if datatype == 'firm+macro+yield':
                        train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
                        train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
                        train_data3 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
                        test_data1 = test_data[['ambnumber','year','country']+factor_list+[predict_label]]
                        test_data2 = test_data[['ambnumber','year','country']+macro_list+[predict_label]]    
                        test_data3 = test_data[['ambnumber','year','country']+yield_list+[predict_label]] 
                        result['SANN0'] = TFFN(test_data1,[4,3,2],test_data2,[4,3,2],test_data3,[4,3,2],datatype,year,predict_label)
                        result['SANN1'] = TFFN2(test_data1,[4,3,2],test_data2,[4,3,2],test_data3,[4,3,2],[3],datatype,year,predict_label)
               
            
                    results = pd.concat([results,result])
                    print(time.time()-t0)
                auc = pd.DataFrame(index=[datatype])
                for method in results.columns[3:]:
                    fpr,tpr,_ = metrics.roc_curve(results[predict_label],results[method])
                    auc[method] = metrics.auc(fpr,tpr)
                Table = pd.concat([Table, auc])
        
        
            Table.to_excel(writer,sheet_name=f'auc{k}')


auc_std = None

with pd.ExcelFile('tables and figures/TableA.1_10AUC.xlsx') as xls:
    sheet_names = xls.sheet_names
    all_results = []
    
    for sheet in sheet_names:
        if sheet.startswith('auc'):  
            df = pd.read_excel(xls, sheet_name=sheet)
            df = df.set_index(df.columns[0])
            all_results.append(df)
            
    auc_std = pd.concat(all_results, axis=0).groupby(level=0).std(ddof=0)

    with pd.ExcelWriter('tables and figures/TableA.1.xlsx') as writer:
        auc_std.to_excel(writer, sheet_name='auc_std')

os.remove('tables and figures/TableA.1_10AUC.xlsx')        
