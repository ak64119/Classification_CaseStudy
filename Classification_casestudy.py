# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:44:32 2019

@author: akshay
"""

# =============================================================================
# CLASSIFYING PERSONAL INCOME
# =============================================================================

#==============================================================================
# Import Required Packages
#==============================================================================

import os
import numpy as np                      # To perform numerical operations
import pandas as pd                     # To work with dataframes
import seaborn as sns                   # To visualize data
from sklearn.model_selection import train_test_split #for partioning the data
from sklearn.linear_model import LogisticRegression #Importing library for logistic regression
from sklearn.metrics import accuracy_score,confusion_matrix #Importing performance metrics- accuracy score and confusion_matrix

#==============================================================================
# Importing Data 
#==============================================================================

# Set Directory
os.chdir('E:/New Volume/Academic/NPTEL/Python_for_DS')

data_income = pd.read_csv("income.csv")

# Creating a copy of original data
data = data_income.copy()

"""
#Exploratory data analysis:

#1.Getting to know the data
#2.Data Preprocessing (Missing values)
#3.Cross tables and data visualizations
"""

#==============================================================================
# Getting to know the data
#==============================================================================
#**** To check variables' data type
print(data.info())

#**** Check for missing values
data.isnull()

print('Data columns with null values:\n', data.isnull().sum())
#**** No missing values !

#**** Summary of numerical variables
summary_num = data.describe()
print(summary_num)

#**** Summary of categorical variables
summary_cate = data.describe(include = "O")
print(summary_cate)

#**** Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

#**** Checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

#**** There exists a '?' instead of nan

"""
Go back and read the data by including "na_values[' ?']" to consider ' ?' as nan

"""

data = pd.read_csv("income.csv", na_values = [" ?"])

#==============================================================================
# Data pre-processing
#==============================================================================
data.isnull().sum()

missing = data[data.isnull().any(axis=1)]
# axis => to consider atleast one column value is missing

"""Points to note:
    1. Missing values in Jobtype = 1809
    2. Missing values in Occupation = 1816
    3. There are 1809 rows where two specific columns, i.e., occupation and 
    JobType have missing values
    4. (1816 - 1809) = 7 => You still have occupation unfilled for these 7 rows.
    Because, Jobtype is Never worked.
"""

data2 = data.dropna(axis=0)

# Relationship between independent variables
correlation = data2.corr()

#==============================================================================
# Cross tables and Data Visualization
#==============================================================================
# Extracting the column names
data2.columns


#==============================================================================
# Gender proportion table:
#==============================================================================
gender = pd.crosstab(index = data2["gender"],
                     columns = 'count',
                     normalize = True)

print(gender)
#==============================================================================
# Gender vs Salary Status:
#==============================================================================
gender_salstat = pd.crosstab(index = data2["gender"],
                     columns = data2["SalStat"],
                     margins = True,
                     normalize = 'index')

print(gender_salstat)
#==============================================================================
# Frequency distribution of Salary Status:
#==============================================================================
SalStat = sns.countplot(data2['SalStat'])

"""

75% of people's salary status is <= 50,000
and 25% of people's salary status is > 50,000

"""

######################### Histogram of Age ####################################
sns.distplot(data2['age'], bins = 10, kde = False)
# People with age 20-45 age are high in frequency

################### Box Plot - Age vs Salary Status ###########################
sns.boxplot('SalStat','age',data =data2)
data2.groupby('SalStat')['age'].median()

### people with 35-50 age are more likely to earn > 50,000 USD p.a.
### people with 25-35 age are more likely to earn <= 50,000 USD p.a.

#==============================================================================
# Exploratory Data Analysis
#==============================================================================

# Visualizing Parameters

# Job Type Vs Salary Status
sns.countplot(y="JobType", hue="SalStat", data=data2)

# Education Vs Salary Status
sns.countplot(y="EdType", hue="SalStat", data=data2)

# Occupation Vs Salary Status
sns.countplot(y="occupation", hue="SalStat", data=data2)

# Capatial Gain
sns.distplot(data2['capitalgain'], bins = 10, kde = False)

# Capatial Loss
sns.distplot(data2['capitalloss'], bins = 10, kde = False)

# Hours per week vs Salary Status
sns.boxplot(x='SalStat',y='hoursperweek', data=data2)

#==============================================================================
# Logistic Regression Model
#==============================================================================
 
# Reindexing the salary status names to 0,1
data2['SalStat'] = data2["SalStat"].map({' less than or equal to 50,000':0,
     ' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2, drop_first=True)  #One hot encoding

# Storing the column names
columns_list = list(new_data.columns)
print(columns_list)

# Separating the input names from data
features = list(set(columns_list)-set(['SalStat']))
print(features)

# Separating the output values in y
y=new_data['SalStat'].values

# Storing the values from input features
x=new_data[features].values

# Splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the model
logistic = LogisticRegression()

# Fitting the value for x and y
logistic.fit(train_x, train_y)

# Prediction from test data
prediction = logistic.predict(test_x)

# Confusion Matrix
confusion_matrix = confusion_matrix(test_y,prediction)
print(confusion_matrix)

# Calculating the accuracy score
accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction).sum())

#==============================================================================
# Logistic Regression - Removing Insignificant Variables
#==============================================================================

# Reindexing the salary status names to 0,1
data2['SalStat'] = data2["SalStat"].map({' less than or equal to 50,000':0,
     ' greater than 50,000':1})
print(data2['SalStat'])

# Removing Insignificant Variables
cols = ['gender','nativecountry','race','JobType']
new_data = data2.drop(cols,axis=1)

new_data=pd.get_dummies(new_data, drop_first=True)  #One hot encoding

# Storing the column names
columns_list = list(new_data.columns)
print(columns_list)

# Separating the input names from data
features = list(set(columns_list)-set(['SalStat']))
print(features)

# Separating the output values in y
y=new_data['SalStat'].values

# Storing the values from input features
x=new_data[features].values

# Splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the model
logistic = LogisticRegression()

# Fitting the value for x and y
logistic.fit(train_x, train_y)

# Prediction from test data
prediction = logistic.predict(test_x)

# Confusion Matrix
confusion_matrix = confusion_matrix(test_y,prediction)
print(confusion_matrix)

# Calculating the accuracy score
accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction).sum())

#==============================================================================
# KNN
#==============================================================================

# importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

# import library for plotting
import matplotlib.pyplot as plt

# Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors=5)

# Fitting the values for X and Y
KNN_classifier.fit(train_x, train_y)

# Predicting the test values with model
prediction = KNN_classifier.predict(test_x)

# Performance metric check
confusion_matrix = confusion_matrix(test_y,prediction)
print("\t","Predicted values")
print("Original values","\n",confusion_matrix)

# Calculating the accuracy score
accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction).sum())

"""
Effect of K value on classifier
"""
Misclassified_sample = []
# Calculating error for K values between 1 and 20
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())
    
print(Misclassified_sample)

#==============================================================================
# END OF SCRIPT
#==============================================================================