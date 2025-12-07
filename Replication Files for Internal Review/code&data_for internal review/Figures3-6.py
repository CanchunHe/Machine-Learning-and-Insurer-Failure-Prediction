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
warnings.filterwarnings('ignore')


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


def Feature_Importance_TFFN(train_data1,test_data1,neurons1,train_data2,test_data2,neurons2,train_data3,test_data3,neurons3,datatype,year,predict_label,ensemble=10,epsilon=0.001):
    X_mean1 = torch.FloatTensor(np.mean(test_data1.iloc[:,3:-1].values,axis=0).reshape(1,-1))
    result1 = np.zeros(X_mean1.shape[1])
    X_mean2 = torch.FloatTensor(np.mean(test_data2.iloc[:,3:-1].values,axis=0).reshape(1,-1))
    result2 = np.zeros(X_mean2.shape[1])
    X_mean3 = torch.FloatTensor(np.mean(test_data3.iloc[:,3:-1].values,axis=0).reshape(1,-1))
    result3 = np.zeros(X_mean3.shape[1])
    for i in range(ensemble):
        with open(r'models_final\TFFN'+str(neurons1)+str(neurons2)+str(neurons3)+'_'+datatype+'_'+str(year)+'_'+str(i)+'_'+predict_label+'.pk','rb') as f:
            select_model = pickle.load(f)
        for j in range(X_mean1.shape[1]):
            temp1 = copy.deepcopy(X_mean1)
            temp1[:,j] += epsilon
            temp2 = copy.deepcopy(X_mean1)
            temp2[:,j] -= epsilon
            result1[j] += min(np.abs((select_model(temp1,X_mean2,X_mean3)[0] - select_model(temp2,X_mean2,X_mean3)[0]).data.numpy()[0]/(2*epsilon)),10)
            # result1[j] += (select_model(temp1,X_mean2,X_mean3)[0] - select_model(temp2,X_mean2,X_mean3)[0]).data.numpy()[0]/(2*epsilon)
        for j in range(X_mean2.shape[1]):
            temp1 = copy.deepcopy(X_mean2)
            temp1[:,j] += epsilon
            temp2 = copy.deepcopy(X_mean2)
            temp2[:,j] -= epsilon
            result2[j] += min(np.abs((select_model(X_mean1,temp1,X_mean3)[0] - select_model(X_mean1,temp2,X_mean3)[0]).data.numpy()[0]/(2*epsilon)),10)
        for j in range(X_mean3.shape[1]):
            temp1 = copy.deepcopy(X_mean3)
            temp1[:,j] += epsilon
            temp2 = copy.deepcopy(X_mean3)
            temp2[:,j] -= epsilon
            result3[j] += min(np.abs((select_model(X_mean1,X_mean2,temp1)[0] - select_model(X_mean1,X_mean2,temp2)[0]).data.numpy()[0]/(2*epsilon)),10)
    return np.hstack((result1,result2,result3))/ensemble

            
# feature importance SANN0(firm+macro+yield) 
if __name__ == "__main__":      
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
    datatype = 'firm+macro+yield'
    result =pd.DataFrame(index=factor_list+macro_list+yield_list)
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
        
        yield_data = yield_datas.loc[yield_datas['year']<=year]
        macro_data = macro_datas.loc[macro_datas['year']<=year]
        macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']] = np.log(macro_data[['GDP','GDP per capita','yearly wage (insurance sector)']])
        macro_data[macro_list] = (macro_data[macro_list] - macro_data[macro_list].mean())/macro_data[macro_list].std()
        
        data = pd.merge(data,yield_data,on=['country','year'],how='left')
        data = pd.merge(data,macro_data,on=['country','year'],how='left')             
        data = data[['ambnumber','year','country']+factor_list+yield_list+macro_list+[predict_label]]
            
        train_data = data.loc[data['year']<year]
        train_data['sample_weight'] = 1
        train_data.loc[train_data[predict_label]==1,'sample_weight'] = train_data.shape[0]/train_data[predict_label].sum() -1   
        test_data = data.loc[data['year']==year]
        
        train_data1 = train_data[['ambnumber','year','country']+factor_list+[predict_label,'sample_weight']]
        train_data2 = train_data[['ambnumber','year','country']+macro_list+[predict_label,'sample_weight']]
        train_data3 = train_data[['ambnumber','year','country']+yield_list+[predict_label,'sample_weight']]
        test_data1 = test_data[['ambnumber','year','country']+factor_list+[predict_label]]
        test_data2 = test_data[['ambnumber','year','country']+macro_list+[predict_label]]    
        test_data3 = test_data[['ambnumber','year','country']+yield_list+[predict_label]] 
        
        result['SANN0_'+str(year)] = Feature_Importance_TFFN(train_data1,test_data1,[4,3,2],train_data2,test_data2,[4,3,2],train_data3,test_data3,[4,3,2],datatype,year,predict_label)
        print(time.time()-t0)

    result = result.mean(axis=1)  
    
    
    
    
    ## Figure3
    Figure3 = pd.DataFrame([result.loc[factor_list].sum()/result.sum(), result.loc[macro_list].sum()/result.sum(), result.loc[yield_list].sum()/result.sum()])
    Figure3.index = ['firm', 'macro', 'yield']
    Figure3.columns = ['relative importance']
    
    
    ## Figure4
    Figure4 = pd.DataFrame([result['mutual']/result[factor_list].sum(), result['composite']/result[factor_list].sum(), result['movingstandarddeviationofrop']/result[factor_list].sum(),
                            result[['profitorlossbeforetax','profitorlossaftertax','rop','roa','roe','roabeforetax','roebeforetax','movingaverageofroa','movingaverageofrop','movingaverageofroe',
                                    'movingaverageofroabeforetax','movingaverageofroebeforetax','riskedadjustedroa','riskedadjustedroe','riskedadjustedroabeforetax',
                                    'riskedadjustedroebeforetax']].sum()/result[factor_list].sum(), 
                            result[['grosspremiumswritten','netpremiumswritten','totalassets','totalliabilities','totaldebtors', 'grosstechnicalreserves','nettechnicalreserves','numberofemployees']].sum()/result[factor_list].sum(), 
                            result[['capitalsurplus','netpremiumswrittencapitalsurplus','nettechnicalreservescapitalsurpl','grosspremiumswrittencapitalsurpl','grosstechnicalreservescapitalsur','totaldebtorscapitalsurplus',
                                    'capitalsurplustotalassets','leverageratio','totaldebtorstotalassets']].sum()/result[factor_list].sum(), 
                            result[['totalinvestments','investmentasset','totalinvestmenttotalliabilities']].sum()/result[factor_list].sum(), 
                            result[['movingstandarddeviationofroa','movingstandarddeviationofroabefo', 'movingstandarddeviationofroe','movingstandarddeviationofroebefo']].sum()/result[factor_list].sum(), 
                            result[['liquidassets','liquidassetsnettechnicalreserves','liquidassetstotalliabilities']].sum()/result[factor_list].sum(),
                            result[['life','affiliated']].sum()/result[factor_list].sum(),
                            result[['changeinasset','changeingrosspremium', 'changeinnetpremium']].sum()/result[factor_list].sum(), 
                            result[['premiumretainratio','nettechnicalreservesgrosstechnic']].sum()/result[factor_list].sum(),
                            result[['numberoflinesofbusiness','numberofpremiumregions']].sum()/result[factor_list].sum()])
    Figure4.index = ['mutual', 'composite', 'moving standard deviation of ROP', 'profitability', 'size', 'capital adequacy', 'investment', 'volatility', 'liquidity', 
                     'company type', 'firm growth', 'reinsurance', 'diversification']
    Figure4.columns = ['relative importance']
    
    
    ## Figure5
    Figure5 = pd.DataFrame([result['change in IMF one year T-bill rate']/result[macro_list].sum(), result['GDP']/result[macro_list].sum(), 
                            result['yearly return of MSCI']/result[macro_list].sum(),                            
                            result[['real GDP growth','change in yearly wage','growth of industry production',
                                    'growth of capital','growth of import','growth of export','growth of consumption per capita','growth of gov expenditure','growth of tax revenue','unemployment rate',
                                    'change in unemployment rate']].sum()/result[macro_list].sum(), 
                            result[['Domestic private credit / GDP','Domestic credit / GDP','insurance density',
                                    'insurance penetration']].sum()/result[macro_list].sum(), 
                            result[['change in population','Labor force / population','growth of population with access to Internet',
                                    'growth of labor in agriculture industry']].sum()/result[macro_list].sum(), 
                            result[['change in current account','Capital Ratio of Banks','growth of carbon emission']].sum()/result[macro_list].sum(),
                            result[['inflation','change in exchange rate']].sum()/result[macro_list].sum(), 
                            result[['yearly wage (insurance sector)','GDP per capita']].sum()/result[macro_list].sum()])
    
    Figure5.index = ['change in IMF one year T-bill rate', 'GDP', 'yearly return of MSCI', 'growth', 'market maturity', 'demographics', 
                     'others', 'financial market indicators', 'general economic conditions']
    Figure5.columns = ['relative importance']
    
    
    ## Figure6
    Figure6 = (result[yield_list]/result[yield_list].sum()).to_frame()
    Figure6.columns = ['relative importance']
    
    #save relative importance results
    with pd.ExcelWriter('tables and figures/Figures3-6.xlsx') as writer:
        Figure3.to_excel(writer,sheet_name='Figure3')
        Figure4.to_excel(writer,sheet_name='Figure4')
        Figure5.to_excel(writer,sheet_name='Figure5')
        Figure6.to_excel(writer,sheet_name='Figure6')