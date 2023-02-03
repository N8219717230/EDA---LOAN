#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("loan.csv")
df


# In[3]:


df.shape


# In[4]:


df.columns


# # Data cleaning
# 
# Some columns have a large number of missing values, let's first fix the missing values and then check for other types of data quality problems.

# In[5]:


df.isnull().sum()


# In[6]:


round(df.isnull().sum()/len(df.index),2)*100


# ou can see that many columns have 100% missing values, some have 65%, 33% etc. First, let's get rid of the columns having 100% missing values.

# # Removing the coloumns having less than 90%

# In[7]:


missing_values =  df.columns[round(df.isnull().sum()/len(df.index),2)*100 > 90]
missing_values


# In[8]:


df1 = df.drop(missing_values , axis = 1)
df1.shape


# In[9]:


round(df1.isnull().sum()/len(df1.index),2)*100


# In[10]:


# There are now 2 columns having approx 33 and 65% missing values - 
# description and months since last delinquent

# let's have a look at a few entries in the columns
df1.loc[:, ['desc', 'mths_since_last_delinq']].head()


# The column description contains the comments the applicant had written while applying for the loan. Although one can use some text analysis techniques to derive new features from this column (such as sentiment, number of positive/negative words etc.), we will not use this column in this analysis.
# 
# Secondly, months since last delinquent represents the number months passed since the person last fell into the 90 DPD group. There is an important reason we shouldn't use this column in analysis - since at the time of loan application, we will not have this data (it gets generated months after the loan has been approved), it cannot be used as a predictor of default at the time of loan approval.
# 
# Thus let's drop the two columns.

# In[11]:


# dropping the two columns
df1 = df1.drop(['desc', 'mths_since_last_delinq'], axis=1)


# In[12]:


round(df1.isnull().sum()/len(df1.index),2)*100


# In[13]:


df1.isnull().sum(axis=1)


# In[14]:


# checking whether some rows have more than 5 missing values
len(df1[df1.isnull().sum(axis=1) > 5])


# In[15]:


df1.info()


# In[16]:


# The column int_rate is character type, let's convert it to float
df1['int_rate'] = df1['int_rate'].apply(lambda x: pd.to_numeric(x.split("%")[0]))


# In[17]:


df1.info()


# In[18]:


# also, lets extract the numeric part from the variable employment length

# first, let's drop the missing values from the column (otherwise the regex code below throws error)
df1 = df1[~df1['emp_length'].isnull()]

import re
df1['emp_length'] = df1['emp_length'].apply(lambda x: re.findall('\d+', str(x))[0])

# convert to numeric
df1["emp_length"].astype("int")


# In[19]:


df1.info()


# #  DATA ANALYSIS

# Let's now move to data analysis. To start with, let's understand the objective of the analysis clearly and identify the variables that we want to consider for analysis.
# 
# The objective is to identify predictors of default so that at the time of loan application, we can use those variables for approval/rejection of the loan. Now, there are broadly three types of variables - 1. those which are related to the applicant (demographic variables such as age, occupation, employment details etc.), 2. loan characteristics (amount of loan, interest rate, purpose of loan etc.) and 3. Customer behaviour variables (those which are generated after the loan is approved such as delinquent 2 years, revolving balance, next payment date etc.).
# 
# Now, the customer behaviour variables are not available at the time of loan application, and thus they cannot be used as predictors for credit approval.
# 
# Thus, going forward, we will use only the other two types of variables.

# In[20]:


behaviour_var =  [
  "delinq_2yrs",
  "earliest_cr_line",
  "inq_last_6mths",
  "open_acc",
  "pub_rec",
  "revol_bal",
  "revol_util",
  "total_acc",
  "out_prncp",
  "out_prncp_inv",
  "total_pymnt",
  "total_pymnt_inv",
  "total_rec_prncp",
  "total_rec_int",
  "total_rec_late_fee",
  "recoveries",
  "collection_recovery_fee",
  "last_pymnt_d",
  "last_pymnt_amnt",
  "last_credit_pull_d",
  "application_type"]
behaviour_var


# In[21]:


df1 = df1.drop(behaviour_var , axis = 1)


# In[22]:


df1.shape


# In[23]:


# also, we will not be able to use the variables zip code, address, state etc.
# the variable 'title' is derived from the variable 'purpose'
# thus let get rid of all these variables as well

df1 = df1.drop(['title', 'url', 'zip_code', 'addr_state'], axis=1)


# In[24]:


df1.shape


# df1.info()

# In[25]:


df1["loan_status"].value_counts()


# You can see that fully paid comprises most of the loans. The ones marked 'current' are neither fully paid not defaulted, so let's get rid of the current loans. Also, let's tag the other two values as 0 or 1.

# In[26]:


# filtering only fully paid or charged-off
df1 = df1[df1['loan_status'] != 'Current']
df1['loan_status'] = df1['loan_status'].apply(lambda x: 0 if x=='Fully Paid' else 1)

# converting loan_status to integer type
df1['loan_status'] = df1['loan_status'].apply(lambda x: pd.to_numeric(x))

# summarising the values
df1['loan_status'].value_counts()


# # Univariate Analysis
# First, let's look at the overall default rate.

# In[27]:


round(np.mean(df1['loan_status']), 2)


# it show that ther is 15% chance of default

# # ### now under stand more by using garphs

# In[28]:


import matplotlib.pyplot as plt
sns.barplot(x="grade", y ="loan_status" , data =df1)
plt.show()


# In[29]:


plt.figure(figsize=(16, 6))
sns.barplot(x="sub_grade", y ="loan_status" , data =df1)

plt.show()


# In[30]:


def plot_cat(cat_var):
    sns.barplot(x=cat_var, y='loan_status', data=df1)
    plt.show()
    


# In[31]:


plot_cat("term")


# In[32]:


plot_cat("home_ownership")


# In[33]:


plot_cat("verification_status")


# In[34]:


df1.info()


# In[35]:


plt.figure(figsize = (20,7))
plot_cat("purpose")


# In[91]:


# let's also observe the distribution of loans across years
# first lets convert the year column into datetime and then extract year and month from it
df1['issue_d'].astype("str")


# In[109]:


from datetime import datetime


df1['issue_d'] = df1['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))


# In[110]:


# extracting month and year from issue_date
df1['month'] = df1['issue_d'].apply(lambda x: x.month)
df1['year'] = df1['issue_d'].apply(lambda x: x.year)


# In[112]:


# let's first observe the number of loans granted across years
df1.groupby('year').year.count()


# In[131]:


# number of loans across months
df1.groupby('month').month.count()


# In[129]:


plot_cat("month")


# In[132]:


plot_cat("year")


# In[41]:


sns.distplot(df1["loan_amnt"])
plt.show


# The easiest way to analyse how default rates vary across continous variables is to bin the variables into discrete categories.
# 
# Let's bin the loan amount variable into small, medium, high, very high.

# In[134]:


# binning loan amount
def loan_amount(n):
    if n < 5000:
        return 'low'
    elif n >=5000 and n < 15000:
        return 'medium'
    elif n >= 15000 and n < 25000:
        return 'high'
    else:
        return 'very high'
        
df1['loan_amnt'] = df1['loan_amnt'].apply(lambda x: loan_amount(x))
        


# In[135]:


df1['loan_amnt'].value_counts()


# In[53]:


# let's also convert funded amount invested to bins
df1['funded_amnt_inv'] = df1['funded_amnt_inv'].apply(lambda x: loan_amount(x))


# In[54]:


# funded amount invested
plot_cat('funded_amnt_inv')


# In[118]:


# lets also convert interest rate to low, medium, high
# binning loan amount
def int_rate(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=15:
        return 'medium'
    else:
        return 'high'
    
    
df1['int_rate'] = df1['int_rate'].apply(lambda x: int_rate(x))


# In[119]:


# comparing default rates across rates of interest
# high interest rates default more, as expected
plot_cat('int_rate')


# In[120]:


# debt to income ratio
def dti(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=20:
        return 'medium'
    else:
        return 'high'
    

df1['dti'] = df1['dti'].apply(lambda x: dti(x))


# In[121]:


plot_cat("dti")


# # comparing default rates across debt to income ratio
# # high dti translates into higher default rates, as expected

# In[ ]:


# funded amount
def funded_amount(n):
    if n <= 5000:
        return 'low'
    elif n > 5000 and n <=15000:
        return 'medium'
    else:
        return 'high'
    
df1['funded_amnt'] = df1['funded_amnt'].apply(lambda x: funded_amount(x))


# In[ ]:


plot_cat("funded_amnt")


# In[45]:


# installment
def installment(n):
    if n <= 200:
        return 'low'
    elif n > 200 and n <=400:
        return 'medium'
    elif n > 400 and n <=600:
        return 'high'
    else:
        return 'very high'
    
df1['installment'] = df1['installment'].apply(lambda x: installment(x))


# In[46]:



plot_cat("installment")

# comparing default rates across installment
# the higher the installment amount, the higher the default rate
# In[39]:


# annual income
def annual_income(n):
    if n <= 50000:
        return 'low'
    elif n > 50000 and n <=100000:
        return 'medium'
    elif n > 100000 and n <=150000:
        return 'high'
    else:
        return 'very high'

df1['annual_inc'] = df1['annual_inc'].apply(lambda x: annual_income(x))


# In[49]:


plot_cat("annual_inc")


# annual income and default rate
# lower the annual income, higher the default rate

# # Segmented Univariate Analysis

# In[47]:


# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(16, 6))
plot_cat('purpose')


# In the upcoming analyses, we will segment the loan applications across the purpose of the loan, since that is a variable affecting many other variables - the type of applicant, interest rate, income, and finally the default rate.
# 

# In[63]:


plt.figure(figsize= (25,7))
sns.countplot(x ="purpose" ,data = df1)
plt.show()


# Let's analyse the top 4 types of loans based on purpose: consolidation, credit card, home improvement and major purchase.

# In[69]:


# filtering the df for the 4 types of loans mentioned above
main_purposes = ["credit_card","debt_consolidation","home_improvement","major_purchase"]
df1 =df1[df1["purpose"].isin(main_purposes)]
df1["purpose"].value_counts()


# In[70]:


sns.countplot(x="purpose", data =df1)
plt.show()


# In[72]:


# let's now compare the default rates across two types of categorical variables
# purpose of loan (constant) and another categorical variable (which changes)

plt.figure(figsize =(16,8))
sns.barplot(x ="term" , y = "loan_status", hue = "purpose" , data = df1)
plt.show()


# In[126]:


def plot_segmented(cat_var):
    plt.figure(figsize =(16 , 8))
    sns.barplot(x = cat_var, y= "loan_status" ,hue = "purpose" ,data =df1 )
    plt.show()


# In[79]:


plot_segmented("term")


# In[80]:


plot_segmented("grade")


# In[95]:


plot_segmented("home_ownership")


# In[96]:


plot_segmented("emp_length")


# In[136]:


plt.figure(figsize =(16,8))
sns.barplot(x ="loan_amnt" , y = "loan_status", hue = "purpose" , data = df1)
plt.show()


# In[122]:


plot_segmented("int_rate")


# In[123]:


# installment
plot_segmented('installment')


# In[124]:


# debt to income ratio
plot_segmented('dti')


# In[125]:


# annual income
plot_segmented('annual_inc')


# 

# In[127]:


plot_segmented("year")


# In[ ]:




