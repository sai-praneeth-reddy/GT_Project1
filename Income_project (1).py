#!/usr/bin/env python
# coding: utf-8

# ### Project 1 Data Analysis
# * Understand the relationship between income and confirmed deaths of corona. 
# 
# -----

# ### Note
# * Instructions have been included for each segment. You do not have to follow them exactly, but they are included to help you think through the steps.

# In[28]:


import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np 
from scipy.stats import linregress
from sklearn import datasets
import statsmodels.api as sm

# File to Load 
file_to_load1 = "income.csv"
file_to_load2 = "us-states.csv"


# In[29]:


(file_to_load1)


# * Display the total number of players
# 

# In[30]:


# Read Purchasing File and store into Pandas data frame


income_data_df = pd.read_csv(file_to_load1)
state_data_df = pd.read_csv(file_to_load2)
combined_df= pd.merge(state_data_df,income_data_df, on = "state")
combined_df['date'].dtype


# In[31]:


## County count
state_count = len(combined_df["state"].unique())
print(state_count)

 #get the range of data dates 
print(min(combined_df['date']))
print(max(combined_df['date']))


# ## Corona Cases Analysis (Total)

# * Run basic calculations to obtain number of unique cases.
# 
# 
# * Create a summary data frame to hold the results
# 
# 
# * Optional: give the displayed data cleaner formatting
# 
# 
# * Display the summary data frame
# 

# In[32]:


#group data by cases confirmed on the most recent day
grouped_date_df = combined_df.groupby(["date"])
date_county_df = combined_df["date"].nunique()

grouped_date_df.head(20)


# ## Cases Confirmed

# * Understand the amount of cases by date and county
# 
# 
# 

# 
# ## Take a look at the total cases by county for the most recent date 

# 
# * Create a summary data frame to hold the data for the most recent confirmed cases of corona on 4/17/20
# 
# 

# In[33]:


max_date_df = combined_df["date"]== (max(combined_df['date']))
max_date_df
recent_date = combined_df[max_date_df]
recent_date.head(20)


# 

# 
# 

# In[34]:


#drop dupes for the recent date data 
plot_data_no_dupes= recent_date.drop_duplicates(subset="state", keep="first")
plot_data_no_dupes


# In[ ]:





# ## Check County Data 

# * Check to make sure county data aligns with state

# In[ ]:





# 

# 
# 
# 

# In[ ]:





# ## Scatter plots

# * Scatter plot for all income v cases 
# 
# 

# In[35]:


case_plot = plot_data_no_dupes.plot.scatter(x='Income',y='deaths',c='Blue')
plt.xlabel('Income per capita 2019')
plt.ylabel('Confirmed Corona deaths')
plt.title('Corona Case Update for April 17th, 2020')
plt.show()


# ## Regression Graphs

# * Use scatter plot to add regression lines 
# 
# 

# In[41]:


income= plot_data_no_dupes['Income']
deaths= plot_data_no_dupes['deaths']
correlation= st.pearsonr(income,deaths)
(slope, intercept, rvalue, pvalue, stderr) = linregress(income,deaths)
regress_values = income * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(income,deaths)
plt.plot(income,regress_values,"r-")
plt.annotate(line_eq,(0,0),fontsize=15,color="red")
plt.xlabel('Average Household Income')
plt.ylabel('Confirmed Corona deaths')
# Print r square value
print(f"R squared: {rvalue**2}")
print(f"The correlation between Average Household Income and confirmed corona deaths is {round(correlation[0],2)}")
plt.show()
plt.savefig('12.png')


# In[37]:


X = plot_data_no_dupes["Income"]
y = plot_data_no_dupes["deaths"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[38]:


# The null hypothesis, which is that income does not affect the death rate has been rejected. 
#Therefore, we can conclude that states with lower average incomes have a greater death rate.


# In[ ]:




