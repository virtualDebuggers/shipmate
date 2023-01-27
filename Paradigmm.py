#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing the libraries 
import streamlit as st #importing streamlit
import pandas as pd # Used to manipulate the dataframe
import numpy as np # Used for scientific calculations
import matplotlib.pyplot as plt # Used for data visualisation
import seaborn as sns # Used for data visualisation
import missingno as msno # Used for visualizing missing values 
import warnings # Used to remove warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split,cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
get_ipython().system('pip install lightgbm')
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import math # Used for mathematical operations
from IPython.display import Image # Used for displaying image
import warnings # Used to remove warnings
warnings.filterwarnings("ignore")
sns.set(rc={"figure.figsize": (20, 15)})
sns.set_style("whitegrid")
from sklearn import linear_model


# In[4]:


st.title('ShipMate')
st.markdown('hasaskba')
st.text('Quantity')


# In[92]:


# Importing the dataset
df = pd.read_excel('Dataset_Hackathon.xlsx')


# In[93]:


# Displaying the dataset
df.head()


# In[94]:


# Importing the distance dataset
df2 = pd.read_csv('distances.csv')


# In[95]:


# Displaying the distance dataset
df.head()


# In[96]:


# Creating a heatmap in the dataframe using viridis colour
df.head(10).style.background_gradient(cmap = "viridis")


# In[97]:


# Frieght cost is the amount paid to a carrier company for the transportation of goods from the point of origin to an agreed location


# In[98]:


# Getting the first five rows of the dataframe
df.head()


# In[99]:


# Getting the last five rows of the dataframe
df.tail()


# In[100]:


# Getting the basic information about the dataframe
df.describe().transpose().style.background_gradient(cmap = "magma")


# In[101]:


# Getting basic information from the dataframe
df.info()


# In[102]:


# Checking if there is duplicate data in the dataframe
df.duplicated().sum()


# In[103]:


# We are keeping the duplicate data


# In[104]:


# Checking if there are any null values in the dataframe
pd.options.display.max_rows= None # Shows all the rows
df.isnull().sum()


# In[105]:


# We observe that there are no null values


# In[106]:


# We move on to check which features correlates with the Predicted Shipment Cost the most


# In[107]:


# Creating a list of column names for categorical and numerical features
cate_feat = list(df.select_dtypes(include = [object]).columns)
num_feat = list(df.select_dtypes(include = [int,float]).columns)

print(cate_feat)
print("\n")
print(num_feat)


# In[108]:


# Heatmap for all the remaining numerical data including the taget 'SalePrice'
# Defining the heatmap parameters
pd.options.display.float_format = "{:,.2f}".format # Round off to two decimal places

# Defining the correlation matrix
corr_matrix = df[num_feat].corr()

# Replacing correlation < |0.3| by 0 for a better visibility
corr_matrix[(corr_matrix < 0.3) & (corr_matrix > -0.3)] = 0

# Plotting the heatmap
sns.heatmap(corr_matrix, vmax=1.0, vmin=-1.0, linewidths=0.1, annot_kws={"size": 9, "color": "black"},annot=True)
plt.title("ShipmentCost Correlation")


# In[109]:


# Dropping the unrelated features
X1 = df.drop(['Country', 'Commodity','Flow','Category'], axis = 1)
X1.head()


# In[110]:


# Plotting a subplot
columns = X1.drop("Frieght Cost (USD)", axis="columns").columns
halfcol = math.ceil(len(columns)/2)
fig = plt.figure(figsize=(10,10))

for i in range(len(columns)):
    plt.subplot(halfcol, halfcol, i+1)
    column = columns[i]
    plt.xlabel(column)
    plt.ylabel("Frieght Cost (USD)")
    plt.scatter(X1[column], X1["Frieght Cost (USD)"])

fig.tight_layout()


# In[134]:


# Performing Linear Regression
x = X1.drop("Frieght Cost (USD)", axis="columns")
y = X1["Frieght Cost (USD)"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
LR = linear_model.LinearRegression()
LR.fit(x_train, y_train)


# In[135]:


# Getting the coefficients
mymodel.coef_


# In[136]:


# Getting the intercept
mymodel.intercept_


# In[137]:


# Predicting the x_test
mymodel.predict(x_test)


# In[138]:


# Getting the y_test
y_test


# In[139]:


# Checking the accuracy
mymodel.score(x_test,y_test)


# In[140]:


# Checking a predicted value
mymodel.predict([[34,12,5940.83050148322]])


# In[141]:


# Checking another predicted value
mymodel.predict([[3,12,5940.83050148322]])


# In[142]:


# Random Forest Regressor


# In[143]:


# Importing the library

from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor()
RF.fit(x_train,y_train) # Fitting the data

x_test_pred_RF = RF.predict(x_test)  #Predicted x_test


# In[144]:


# Displaying the predicted x_test
x_test_pred_RF


# In[145]:


y_test   # y test


# In[146]:


x_train_pred_RF = LR.predict(x_train)  # Predicted x_train
x_train_pred_RF 


# In[147]:


y_train  # y train


# In[148]:


# Random Foresr Regressor Score


# In[149]:


print('Training score for Random Forest Regressor is',RF.score(x_train,y_train))


# In[ ]:





# In[ ]:




