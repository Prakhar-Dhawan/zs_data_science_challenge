# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 17:48:55 2018

@author: Prakhar
"""
#LOADING DATASET
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd# -*- coding: utf-8 -*-
from datetime import timedelta 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
from pandas.tools.plotting import autocorrelation_plot

train = pd.read_csv('yds_train2018.csv')
train.shape
train.info()
train.columns

#EXPLORATION
'''
train.Sales.value_counts()
train['Year'] = pd.Categorical(train.Year)
train['Month'] = pd.Categorical(train.Month)
train['Week'] = pd.Categorical(train.Week)
train['Merchant_ID'] = pd.Categorical(train.Merchant_ID)
train['Product_ID'] = pd.Categorical(train.Product_ID)
train['Country'] = pd.Categorical(train.Country)

#univariate
train['Year'].value_counts().plot.bar(title='Year')
train['Month'].value_counts().plot.bar(title='Month')
train['Week'].value_counts().plot.bar(title='Week')
train['Product_ID'].value_counts().plot.bar(title='Product_ID')
train['Country'].value_counts().plot.bar(title='Country')

sns.distplot(train['Sales'])
train['Sales'].plot.box(figsize=(16,5))

#bivariate
train.boxplot(column='Sales', by='Year')
train.boxplot(column='Sales', by='Month')
train.boxplot(column='Sales', by='Week')
train.boxplot(column='Sales', by='Merchant_ID')
train.boxplot(column='Sales', by='Product_ID')
train.boxplot(column='Sales', by='Country')

#anova
model1 = smf.ols(formula='Sales ~ C(Country)', data=train)
results1 = model1.fit()
print(results1.summary())

test_3 = train[['Sales','Country']].dropna()
m1 = test_3.groupby('Country').mean()
sd1 = test_3.groupby('Country').std()
print(m1)
print(sd1)

mc1 = multi.MultiComparison(train['Sales'], train['Country'])
res1 = mc1.tukeyhsd()
print(res1.summary())
'''
#impuating missing values of Sales

b1 = train.groupby('Merchant_ID')['Sales'].count()

train_1 = train[train['Sales']==0]
train_1.shape
c1 = train_1.groupby('Merchant_ID')['Sales'].count()

d1 = c1/b1
d1 = d1[d1==1]
d1.head()
d1.shape

z1 = d1.index
z1.shape

trein_1 = train[-train['Merchant_ID'].isin(z1)]

trein_1['Sales'][trein_1['Sales']==0] = np.NaN
trein_1.isnull().sum()

def impute_mean(series):
    return series.fillna(series.mean())

trein_1['Sales'] = trein_1.groupby(['Merchant_ID','Product_ID']).Sales.transform(impute_mean)
trein_1.Sales.value_counts()
trein_1.isnull().sum()
trein_1.shape
#trein_1.to_csv('train_final.csv')

train_final = trein_1.dropna()
train_final.shape
train_final.Sales.value_counts()

#preparing train data for modelling
train_final.columns
train_final1 = train_final.pivot_table(index=['Year','Month','Country'],columns='Product_ID', values='Sales', aggfunc= 'sum')
train_final1.shape
train_final1.columns
#train_final1.to_csv('train_final_pivot.csv')

train_finally = train_final1.stack(level = 'Product_ID' )
#train_finally = pd.melt(train_final1, id_vars = ['Count'],var_name=[1,2,3,4,5], value_name='Count1')
train_finally.shape
#train_finally.columns = ['Year','Month','Product_ID','Sales']
#train_finally.to_csv('train_finally.csv')


train_mdl = pd.DataFrame(train_finally)
train_mdl.reset_index(level=['Year','Month','Product_ID','Country'], inplace=True)
train_mdl.columns = ['Year','Month','Country','Product_ID','Sales']
train_mdl.shape
#train_mdl.to_csv('Train_mdl.csv')
train_mdl.info()

test_mdl = pd.read_csv('yds_test2018.csv')

exp = pd.read_csv('promotional_expense.csv')
exp.shape
exp.info()
exp.columns

exp = exp.rename(columns = {'Product_Type' : 'Product_ID'})

data = pd.concat([train_mdl, test_mdl])
data.shape
data.info()
data.columns

data_1 = pd.merge(left=data, right=exp, on = ['Year', 'Month', 'Product_ID', 'Country'], how='left' )
data_1.info()
data_1.columns
data_1.shape

data_1.isna().sum()

#PREPARING HOLIDAY DATA
dum = pd.ExcelFile('holidays.xlsx')
holid = dum.parse(0)

holid.shape
holid.columns
holid.info()
holid['Date'] = pd.to_datetime(holid['Date'])
holid['Month'] = holid['Date'].dt.month
holid['Year'] = holid['Date'].dt.year

years = [2013,2014,2015,2016,2017]
holid = holid[holid['Year'].isin(years)]
holid_1 = holid.groupby(['Year','Month','Country'])['Holiday'].count()
holid_1.shape
holid_1 = pd.DataFrame(holid_1)
#train_mdl.reset_index(level=['Year','Month','Product_ID','Country'], inplace=True)
holid_1.reset_index(level=['Year','Month','Country'],inplace=True)
holid_1.shape
holid_1.columns
#holid_1.to_csv('holidays final.csv')
holid_1.info()

#PREPARING FINAL TRAINING DATA
#BY MERGING THE HOLIDAY DATA

data_2 = pd.merge(left=data_1, right=holid_1, on = ['Year', 'Month', 'Country'], how='left' )
data_2.shape
data_2.columns
data_2.info()
data_2['Holiday'][data_2['Holiday'].isnull()] = 0
data_2.Holiday.value_counts()
#data_2.to_csv('Train_model.csv')

#ÉXPLORATION OF PREPARED FINAL TRAINING DATA'''
'''
data_2.isnull().sum()
data_2['Year'] = pd.Categorical(data_2.Year)
data_2['Country'] = pd.Categorical(data_2.Country)
data_2['Month'] = pd.Categorical(data_2.Month)
data_2['Product_ID'] = pd.Categorical(data_2.Product_ID)
#univariate
data_2['Year'].value_counts().plot.bar(title='Year')
data_2['Country'].value_counts().plot.bar(title='Country')
data_2['Month'].value_counts().plot.bar(title='Month')
data_2['Product_ID'].value_counts().plot.bar(title='Product_ID')

data_2.Expense_Price.describe()
sns.distplot(data_2['Expense_Price'].dropna())
(data_2['Expense_Price'].dropna()).plot.box(figsize=(16,5))
sns.distplot(np.log(data_2['Expense_Price'].dropna()))
(np.log(data_2['Expense_Price'].dropna())).plot.box(figsize=(16,5))

data_2.Sales.describe()
sns.distplot(data_2['Sales'].dropna())
(data_2['Sales'].dropna()).plot.box(figsize=(16,5))
sns.distplot(np.log(data_2['Sales'].dropna()))
(np.log(data_2['Sales'].dropna())).plot.box(figsize=(16,5))

data_2.Holiday.describe()
sns.distplot(data_2['Holiday'])
data_2['Holiday'].plot.box(figsize=(16,5))
sns.distplot(np.log(data_2['Holiday']+1))

#bivariate
data_2.boxplot(column='Expense_Price', by='Year')
data_2.boxplot(column='Expense_Price', by='Month')
data_2.boxplot(column='Expense_Price', by='Product_ID')
data_2.boxplot(column='Expense_Price', by='Country')
data_2.boxplot(column='Sales', by='Year')
data_2.boxplot(column='Sales', by='Month')
data_2.boxplot(column='Sales', by='Product_ID')
data_2.boxplot(column='Sales', by='Country')


#ANOVA
model1 = smf.ols(formula='Sales ~ C(Country)', data=data_2)
results1 = model1.fit()
print(results1.summary())

test_3 = data_2[['Sales','Country']].dropna()
m1 = test_3.groupby('Country').mean()
sd1 = test_3.groupby('Country').std()
print(m1)
print(sd1)

mc1 = multi.MultiComparison(test_3['Sales'], test_3['Product_ID'])
res1 = mc1.tukeyhsd()
print(res1.summary())

#cor
df = data_2[['Sales','Expense_Price']]
df.shape
df = df.dropna()
data_2.plot(x='Expense_Price', y ='Sales',kind='scatter')
np.corrcoef(df['Expense_Price'],df['Sales'])
'''
#missing value imputation  - mean
def impute_median(series):
    return series.fillna(series.median())

data_2['Expense_Price'] = data_2.groupby('Country').Expense_Price.transform(impute_median)

data_2.isnull().sum()



#data_2.to_csv('Train_imputed.csv')

#outlier treatment

#MODELING'''
data_2.info()
#Creating GDP per capita column
data_2['GDPPC'] = np.NaN
#df1 = df[(df.a != -1) & (df.b != -1)]
for rowNum in range(0, data_2.shape[0]) : 
        if( (data_2.at[rowNum, 'Country'] == 'Argentina') and (data_2.at[rowNum, 'Year'] ==2013)  ):
            data_2.at[rowNum, 'GDPPC']= 12976
        elif( (data_2.at[rowNum, 'Country'] == 'Argentina') and (data_2.at[rowNum, 'Year'] ==2014)  ):
            data_2.at[rowNum, 'GDPPC']= 12245
        elif( (data_2.at[rowNum, 'Country'] == 'Argentina') and (data_2.at[rowNum, 'Year'] ==2015)  ):
            data_2.at[rowNum, 'GDPPC']= 13467
        elif( (data_2.at[rowNum, 'Country'] == 'Argentina') and (data_2.at[rowNum, 'Year'] ==2016)  ):
            data_2.at[rowNum, 'GDPPC']= 12449
        elif( (data_2.at[rowNum, 'Country'] == 'Argentina') and (data_2.at[rowNum, 'Year'] ==2017)  ):
            data_2.at[rowNum, 'GDPPC']= 14408
        elif( (data_2.at[rowNum, 'Country'] == 'Belgium') and (data_2.at[rowNum, 'Year'] ==2013)  ):
            data_2.at[rowNum, 'GDPPC']= 44209
        elif( (data_2.at[rowNum, 'Country'] == 'Belgium') and (data_2.at[rowNum, 'Year'] ==2014)  ):
            data_2.at[rowNum, 'GDPPC']= 44676
        elif( (data_2.at[rowNum, 'Country'] == 'Belgium') and (data_2.at[rowNum, 'Year'] ==2015)  ):
            data_2.at[rowNum, 'GDPPC']= 45052
        elif( (data_2.at[rowNum, 'Country'] == 'Belgium') and (data_2.at[rowNum, 'Year'] ==2016)  ):
            data_2.at[rowNum, 'GDPPC']= 45457
        elif( (data_2.at[rowNum, 'Country'] == 'Belgium') and (data_2.at[rowNum, 'Year'] ==2017)  ):
            data_2.at[rowNum, 'GDPPC']= 46078
        elif( (data_2.at[rowNum, 'Country'] == 'Columbia') and (data_2.at[rowNum, 'Year'] ==2013)  ):
            data_2.at[rowNum, 'GDPPC']= 7051
        elif( (data_2.at[rowNum, 'Country'] == 'Columbia') and (data_2.at[rowNum, 'Year'] ==2014)  ):
            data_2.at[rowNum, 'GDPPC']= 7291
        elif( (data_2.at[rowNum, 'Country'] == 'Columbia') and (data_2.at[rowNum, 'Year'] ==2015)  ):
            data_2.at[rowNum, 'GDPPC']= 7446
        elif( (data_2.at[rowNum, 'Country'] == 'Columbia') and (data_2.at[rowNum, 'Year'] ==2016)  ):
            data_2.at[rowNum, 'GDPPC']= 7531
        elif( (data_2.at[rowNum, 'Country'] == 'Columbia') and (data_2.at[rowNum, 'Year'] ==2017)  ):
            data_2.at[rowNum, 'GDPPC']= 7600
        elif( (data_2.at[rowNum, 'Country'] == 'Denmark') and (data_2.at[rowNum, 'Year'] ==2013)  ):
            data_2.at[rowNum, 'GDPPC']= 58788
        elif( (data_2.at[rowNum, 'Country'] == 'Denmark') and (data_2.at[rowNum, 'Year'] ==2014)  ):
            data_2.at[rowNum, 'GDPPC']= 59437
        elif( (data_2.at[rowNum, 'Country'] == 'Denmark') and (data_2.at[rowNum, 'Year'] ==2015)  ):
            data_2.at[rowNum, 'GDPPC']= 59967
        elif( (data_2.at[rowNum, 'Country'] == 'Denmark') and (data_2.at[rowNum, 'Year'] ==2016)  ):
            data_2.at[rowNum, 'GDPPC']= 60670
        elif( (data_2.at[rowNum, 'Country'] == 'Denmark') and (data_2.at[rowNum, 'Year'] ==2017)  ):
            data_2.at[rowNum, 'GDPPC']= 61582
        elif( (data_2.at[rowNum, 'Country'] == 'England') and (data_2.at[rowNum, 'Year'] ==2013)  ):
            data_2.at[rowNum, 'GDPPC']= 39996
        elif( (data_2.at[rowNum, 'Country'] == 'England') and (data_2.at[rowNum, 'Year'] ==2014)  ):
            data_2.at[rowNum, 'GDPPC']= 40908
        elif( (data_2.at[rowNum, 'Country'] == 'England') and (data_2.at[rowNum, 'Year'] ==2015)  ):
            data_2.at[rowNum, 'GDPPC']= 41536
        elif( (data_2.at[rowNum, 'Country'] == 'England') and (data_2.at[rowNum, 'Year'] ==2016)  ):
            data_2.at[rowNum, 'GDPPC']= 42039
        elif( (data_2.at[rowNum, 'Country'] == 'Finland') and (data_2.at[rowNum, 'Year'] ==2013)  ):
            data_2.at[rowNum, 'GDPPC']= 45715
        elif( (data_2.at[rowNum, 'Country'] == 'Finland') and (data_2.at[rowNum, 'Year'] ==2014)  ):
            data_2.at[rowNum, 'GDPPC']= 45239
        elif( (data_2.at[rowNum, 'Country'] == 'Finland') and (data_2.at[rowNum, 'Year'] ==2015)  ):
            data_2.at[rowNum, 'GDPPC']= 45151
        elif( (data_2.at[rowNum, 'Country'] == 'Finland') and (data_2.at[rowNum, 'Year'] ==2016)  ):
            data_2.at[rowNum, 'GDPPC']= 45983

#data_2.to_csv('train + Gdp.csv') 
'''
data_2.GDPPC.describe()
sns.distplot(data_2['GDPPC'])
data_2['GDPPC'].plot.box(figsize=(16,5))
sns.distplot(np.log(data_2['GDPPC']))
'''


data_2['Country'] = data_2['Country'].astype(object)
data_2['Country'].value_counts()
data_2['Country'][data_2['Country']=='Argentina'] = 0 
data_2['Country'][data_2['Country']=='Belgium'] = 1 
data_2['Country'][data_2['Country']=='Columbia'] = 2
data_2['Country'][data_2['Country']=='Denmark'] = 3
data_2['Country'][data_2['Country']=='England'] = 4
data_2['Country'][data_2['Country']=='Finland'] = 5
data_2['Country'] = pd.Categorical(data_2.Country)
data_2['Month'] = pd.Categorical(data_2.Month)
data_2['Product_ID'] = pd.Categorical(data_2.Product_ID)
data_2['Year'] = pd.Categorical(data_2.Year)

#Preparing X, X_ and y for modeling

#many outliers are present in the continuous variables Sales and Expense_price
#log transform Sales and Expense_Price to remove the outliers


data_3 = data_2
del data_3['S_No']
data_3.info()
X = data_3.dropna()
y = X['Sales']
X =X.drop('Sales',axis=1)
y.shape
y = np.log(y)
X['Expense_Price'] = np.log(X['Expense_Price'])
X.shape
X.info()
X_ = data_3[pd.isnull(data_3).any(axis=1)]
del X_['Sales']
X_.shape
X_.info()
#X.to_csv('Xzs.csv')
#y.to_csv('Yzs.csv')

X_['Expense_Price'] = np.log(X_['Expense_Price'])


#XGBOOST
X.info()
X_.info()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

X = X.apply(pd.to_numeric)
y = y.apply(pd.to_numeric)
X_ = X_.apply(pd.to_numeric)
del X['Country']
del X_['Country']


import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
xg_reg = xgb.XGBRegressor(objective='reg:linear',seed=123)
xg_reg.fit(X_train, y_train)
preds_xg = xg_reg.predict(X_test)
xg_reg.score(X_test, y_test)
preds_xg.shape

def mean_absolute_percentage_error(y_test, predicted_tb): 
    y_test, predicted_tb = np.array(y_test), np.array(predicted_tb)
    return np.mean(np.abs(y_test - predicted_tb) / (np.abs(y_test) + np.abs(predicted_tb))) * 100

mean_absolute_percentage_error(np.exp(y_test), np.exp(preds_xg))

#crossvalidating
from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(xg_reg, X, y, cv=10))

xg_reg.fit(X, y)
#xgb.plot_importance(xg_reg)

#tuning
from sklearn.model_selection import RandomizedSearchCV
gbm_param_grid = {
            'learning_rate': [0.001,0.01,0.1,1],      
            'n_estimators':[50,100,200,500,1000],       
            'subsample': [0.3,0.5,0.9],              
            'max_depth':[5,20,50],                  
            'colsample_bytree':[0.5,0.8]}           
gbm = xgb.XGBRegressor()
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions= gbm_param_grid, scoring='neg_mean_absolute_error', cv=4, verbose=5, random_state=123, n_iter=150)
#randomized_mse.fit(X, y)
#randomized_mse.best_params_
#randomized_mse.best_score_

from sklearn.model_selection import GridSearchCV
dmatrix = xgb.DMatrix(data= X,label= y)
gbm_matrix_grid = {'learning_rate': [0.05,0.1,0.15],   
                   'n_estimators':[75,100,125,150],         
                   'subsample':[0.4,0.5,0.6],        
                   'max_depth':[30,50,70,90,110,150],          
                   'colsample_bytree':[0.8,1.0] }      
gbm_matrix_grid_1 = {'gamma':[0.00001,0.0001,0.0005,0.001,0.005,0], 'reg_alpha':[0.005,0.01,0.05], 'reg_lambda':[0.005,0.01,0.05] }
gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator = gbm, param_grid = gbm_matrix_grid_1, scoring = 'neg_mean_absolute_error', cv=4, verbose=1)
#grid_mse.fit(X, y)
#grid_mse.best_params_
#grid_mse.best_score_

#fitting final model
xg_reg_fi = xgb.XGBRegressor(objective='reg:linear',seed=123, colsample_bytree=1.0, learning_rate=0.1, max_depth=30, n_estimators=125, subsample=0.5, gamma=0.0005, reg_alpha=0.01, reg_lambda=0.005)
xg_reg_fi.fit(X, y)
preds_xg_fi = xg_reg_fi.predict(X_)
preds_xg_fi.shape
test_mdl['Sales'] = np.exp(preds_xg_fi)
#test_mdl.to_csv('XG_finale.csv')
