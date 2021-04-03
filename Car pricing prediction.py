#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing Libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


os.getcwd()
os.chdir(r'C:\Users\Rashmi\Desktop\Projects\Data - Car Price Analysis\data')
os.getcwd()


# In[4]:


cars = pd.read_csv('automobile.csv',header= None)

cars.head()


# In[5]:


cars.columns = ['Symboling','Normalized_Losses','Company_Name','Fuel_Type','Aspiration','No_of_Door','Body_Style','Drive_wheels','Engine_Location','Wheel_Base','Length','Width','Height','Curb_Weight','Engine_Type','No_of_cylinders','Engine_size','Fuel_system','Bore','Stroke','Compression_Ratio','Horse_Power','Peak_RPM','City_MPG','Highway_MPG','Price']
cars


# In[6]:


print(cars.shape)
print(cars.describe())


# In[7]:


cars.info()


# In[8]:


# Checking for duplicates

cars.loc[cars.duplicated()]


# In[9]:


# Checking Null Values

cars.isnull().sum()


# In[10]:


cars = cars.replace('?',np.NaN)
cars


# In[11]:


##inspecting the missing value again

print('Total NaN: '+ str(cars.isnull().values.sum()))
print('NaN by column: ' '\n')
print(cars.isnull().sum())


# In[12]:


cars.columns


# In[48]:


## Filling NaN values for numbers with Mean

#cars.fillna(cars.median(), inplace=True)
#cars


# In[13]:


## Checking the null values again:

print('Total NaN: '+ str(cars.isnull().values.sum()))
print('NaN by column: ' '\n')
print(cars.isnull().sum())


# In[14]:


## ITERATE OVER EACH COLUMN OF cars

for col in cars:
    ## CHECK IF THE COLUMN IS OF OBJECT TYPE
    if cars[col].dtypes == 'object':
        ## IMPUTE WITH THE MOST FREQUENT VALUES
        cars = cars.fillna(cars[col].value_counts().index[0])
##COUNT THE NO. OF NANs AND PRINT THE COUNT TO VERIFY
print('Total NaN: '+ str(cars.isnull().values.sum()))
print('NaN by column: ' '\n')
print(cars.isnull().sum()) 


# In[15]:


cars.info()


# In[16]:


cars['Price'] = cars.Price.astype('int64')
cars['Horse_Power'] = cars.Horse_Power.astype('int64')
cars['Peak_RPM'] = cars.Peak_RPM.astype('int64')
cars['Normalized_Losses'] = cars.Normalized_Losses.astype('int64')
cars['Bore'] = cars.Bore.astype('float64')
cars['Stroke'] = cars.Stroke.astype('float64')


# In[17]:


cars.dtypes


# In[18]:


#### Analyzing categorical data
#.Company Name
#.Symboling 
#.Fuel type
#.Engine type
#.car body
#.door number
#.engine location
#.fuel system
#. No of cylinders
#.Aspiration
#.Drive wheels

cars['Company_Name'].value_counts()


# In[19]:


## Fuel Type counts

cars['Fuel_Type'].value_counts()


# In[20]:


## No of doors

cars['No_of_Door'].value_counts()


# In[21]:


def convert_number(x):
    return x.map({'two':2,'three':3,'four':4,'five':5,'six':6,'eight':8,'twelve':12})
cars['No_of_Door'] = cars[['No_of_Door']].apply(convert_number)


# In[22]:


cars['No_of_cylinders'].value_counts()
cars['No_of_cylinders'] = cars[['No_of_cylinders']].apply(convert_number)


# In[23]:


## No of doors

cars['No_of_Door'].value_counts()


# In[24]:


cars['No_of_cylinders'].value_counts()


# In[27]:


## Aspiration used in car

cars['Aspiration'].value_counts()


# In[26]:


cars['Body_Style'].value_counts()


# In[28]:


cars['Drive_wheels'].value_counts()


# In[29]:


cars['Engine_Location'].value_counts()


# In[30]:


cars['Wheel_Base'].value_counts().head()


# In[31]:


sns.distplot(cars['Wheel_Base'])


# In[32]:


cars['Length'].value_counts()


# In[33]:


sns.distplot(cars['Length'])


# In[34]:


cars['Engine_Type'].value_counts()


# In[35]:


cars['Fuel_system'].value_counts()


# In[36]:


cars.info()


# In[37]:


cars_numeric = cars.select_dtypes(include=['int64','float64'])
cars_numeric


# In[38]:


plt.figure(figsize =(30,30))
sns.pairplot(cars_numeric)
plt.show()


# In[39]:


plt.figure(figsize=(20,12))
sns.heatmap(cars_numeric.corr(),annot=True)


# In[40]:


cars_numeric.corr()


# In[ ]:


## conclusion
#. Price is highly correlated with wheel_base, length,width, curb_weight,engine size. horse power
#. Independent variables whic are highly correlated: wheel base, length,weight, engine size... all are positively correlated.


# In[41]:


categorical_cols = cars.select_dtypes(include=['object'])
categorical_cols.head()


# In[42]:


plt.figure(figsize=(20,12))
plt.subplot(3,3,1)
sns.boxplot(x= 'Aspiration', y = 'Price', data = cars)
plt.subplot(3,3,2)
sns.boxplot(x= 'Fuel_Type', y = 'Price', data = cars)
plt.subplot(3,3,3)
sns.boxplot(x= 'No_of_Door', y = 'Price', data = cars)
plt.subplot(3,3,4)
sns.boxplot(x= 'Body_Style', y = 'Price', data = cars)
plt.subplot(3,3,5)
sns.boxplot(x= 'Drive_wheels', y = 'Price', data = cars)
plt.subplot(3,3,6)
sns.boxplot(x= 'Engine_Location', y = 'Price', data = cars)


# In[43]:


plt.figure(figsize=(18,10))
sns.boxplot(x= 'Engine_Type', y = 'Price', data = cars)


# In[44]:


plt.figure(figsize=(20,12))
sns.boxplot(x= 'Fuel_system', y = 'Price', data = cars)


# In[45]:


plt.figure(figsize=(14,8))
sns.boxplot(x= 'Body_Style', y = 'Price', data = cars)


# In[46]:


plt.figure(figsize=(22,14))
sns.boxplot(x= 'Company_Name', y = 'Price', data = cars)


# In[ ]:


## Conclusion::
#. Most expensive vehicle belongs to BMW, Jaguar, Porsche
#. Lower priced cars belong to chevrolet
#. The median price of gas vehicles is lower than that of diesel vehicles.
#. 75th percentile of standars aspirated vehicles have a price lower than the median price of turbo aspirated vehicles
#. Two and Four door vehicles are equally priced,, however there atre soem outliers in two door vehicles
#. Hatchback vehicles have the lowest median price , whereas hard top vehicles have the highest median price
#. The price of rear placed engine vehicles is significantly higher than the price of vehicles with front placed engines
#. The median cost of 8 cylinder vehicles is higher than other cylinder categories
#. Vehicles having MPFI fuel system have the highest median price
#. Vehicles with OHCV engine type falls under higher price range


# In[47]:


#### Data Preparation::

# Creating Dummies

cars_dummies = pd.get_dummies(categorical_cols, drop_first= True)
cars_dummies.head()


# In[48]:


cars_dummies.columns


# In[49]:


cars_df = pd.concat([cars,cars_dummies], axis =1)


# In[50]:


cars_df = cars_df.drop(['Fuel_Type','Aspiration','Body_Style','Drive_wheels','Engine_Location','Engine_Type','Fuel_system','Company_Name'], axis=1)


# In[51]:


cars_df.info()


# In[52]:


### Splitting the data into train-test::

from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(cars_df, train_size = 0.8, test_size = 0.2, random_state= 100)


# In[53]:


df_train.shape


# In[54]:


df_test.shape


# In[55]:


### Scaling the data

cars_numeric.columns


# In[56]:


col_list = ['Symboling', 'Normalized_Losses', 'Wheel_Base', 'Length', 'Width',
       'Height', 'Curb_Weight', 'Engine_size', 'Bore', 'Stroke',
       'Compression_Ratio', 'Horse_Power', 'Peak_RPM', 'City_MPG',
       'Highway_MPG', 'Price']


# In[57]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
df_train[col_list] = sc_x.fit_transform(df_train[col_list])


# In[58]:


df_train.describe


# In[59]:


### Model Building::
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
y_train = df_train.pop('Price')
x_train = df_train


# In[60]:


lr.fit(x_train,y_train)


# In[87]:


lr.predict(x_test)


# In[81]:


from sklearn.feature_selection import RFE

## subsetting training data for 15 selected columns

rfe = RFE(lr,15)
rfe.fit(x_train,y_train)


# In[93]:


list(zip(x_train.columns, rfe.support_,rfe.ranking_))


# In[92]:


cols = x_train.columns[rfe.support_]
cols


# In[86]:


### Model 

import statsmodels.api as sm


# In[94]:


## Model 1:

x1 = x_train[cols]
x1_sm = sm.add_constant(x1)
lr_1 = sm.OLS(y_train,x1_sm).fit()


# In[95]:


print(lr_1.summary())


# In[101]:


### VIF(Variance Inflation Factor)::

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['features'] = x1.columns
vif['VIF'] = [variance_inflation_factor(x1.values,i) for i in range(x1.shape[1])]
vif = vif.sort_values(by = 'VIF', ascending = False)
vif['VIF'] = round(vif['VIF'],2)
vif


# In[ ]:


## Conclusion::
#. Here are a few variables which have a large VIF, so those variables are not of use.. and manually removing these variables take time,, so now lets build a model with 10 variables


# In[102]:


## Building model with 10 variables:
lr2 = LinearRegression()
rfe2 = RFE(lr2,10)
rfe2.fit(x_train,y_train)


# In[103]:


list(zip(x_train.columns,rfe2.support_,rfe2.ranking_))


# In[104]:


supported_cols = x_train.columns[rfe2.support_]
supported_cols


# In[106]:


x2 = x_train[supported_cols]
x2_sm = sm.add_constant(x2)
model_2 = sm.OLS(y_train,x2_sm).fit()


# In[107]:


print(model_2.summary())


# In[108]:


### VIF(Variance Inflation Factor)::

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['features'] = x2.columns
vif['VIF'] = [variance_inflation_factor(x2.values,i) for i in range(x2.shape[1])]
vif = vif.sort_values(by = 'VIF', ascending = False)
vif['VIF'] = round(vif['VIF'],2)
vif


# In[111]:


## Manually lets drop the columns wiyth VIF >5  and create the model

#Model 3:

x3=x2.drop(['Engine_Type_rotor'],axis=1)
x3_sm = sm.add_constant(x3)
Model_3 = sm.OLS(y_train,x3_sm).fit()
print(Model_3.summary())


# In[112]:


### VIF(Variance Inflation Factor)::

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['features'] = x3.columns
vif['VIF'] = [variance_inflation_factor(x3.values,i) for i in range(x3.shape[1])]
vif = vif.sort_values(by = 'VIF', ascending = False)
vif['VIF'] = round(vif['VIF'],2)
vif


# In[ ]:


## conclusion::
#.All the VIF values seems to be in a good range and adjusted R2 is also 85.6 %..


# In[114]:


## Making Predictions:

#df_test[col_list] = sc_x.transform(df_test[col_list])


# In[116]:


final_cols = x3.columns


# In[120]:


x_test_model_3 = x_test[final_cols]
x_test_model_3.head()


# In[121]:


x_test_sm = sm.add_constant(x_test_model_3)


# In[122]:


y_pred = Model_3.predict(x_test_sm)


# In[123]:


y_pred.head()


# In[126]:


plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')


# In[127]:


## R squarrec value

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:


# Conclusion:--- Linear equation for price can be given as::
# Price = -0.0803+ 0.7206*Engine_Size + 0.1742*Stroke + 1.14*Company_Name_bmw - 0.6917*Company_Name_isuzu + 1.0433*Company_Name_mercedez-benz + 0.8553*Company_Name_Porsche + 0.5415*Company_Name_saab + 0.9575*Engine_Location_rear - 3.7728*Engine_type_dohcv



