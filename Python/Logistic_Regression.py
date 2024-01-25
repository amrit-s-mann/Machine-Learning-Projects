'''     Import libraries and define path    '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

path = "Data-Science-Course/Sample data/"
dcredit = pd.read_excel(path + 'Helper_Data.xlsx', sheet_name = 'Credit Profile')
print(dcredit)

'''     Logistic Regression modelling    '''

# Defining the dependent and independent variables
Xtrain = dcredit[['Annual_Income', 'Education_Years', 'Age']]
Ytrain = dcredit[['Credit_Profile']]

# Building the model and fitting the data
logreg = sm.Logit(Ytrain, Xtrain).fit()
print(logreg.summary())

'''     Testing the model    '''

# Providing out of sample data for testing and converting to Dataframe object
xdata = {'Annual_Income':[110000,42000,95000,90000,100000],'Education_Years':[10,10,16,20,30],'Age':[30,27,28,40,50]}
ydata = {'Credit_Profile':[1,0,1,1,0]}
xtest = pd.DataFrame(xdata)
ytest = pd.DataFrame(ydata)

# Performing predictions on the test dataset
ypredict = logreg.predict(xtest)
print(ypredict)
predictions = list(map(round, ypredict))

print("Actual values: ", list(ytest.Credit_Profile))
print("Predicted values: ", predictions)

'''     Obtaining Confusion Matrix & Accuracy Score    '''

from sklearn.metrics import (confusion_matrix, accuracy_score)

# Confusion Matrix
cm = confusion_matrix(ytest.Credit_Profile, predictions)
print("Confusion Matrix: \n", cm)

# Accuracy Score
score = accuracy_score(ytest.Credit_Profile, predictions)
print("Accuracy Score: ", score)