# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm                         
from sklearn.linear_model import LinearRegression

# Load excel file
db = pd.read_excel("Data_science_course/Data/Inflation.xlsx", sheet_name = 'Inflation')

#Check content
print(db)

# Assign X and Y from columns in the dataframe
X = db.Money_Supply                                 # Column with values of independent variable
Y = db.Inflation                                    # Column with values of dependent variable
corr = X.corr(Y)
print("Correlation: " + str(round(corr, 3)))
rsquared = corr ** 2
print("R-Squared: " + str(round(rsquared, 3)))
X = sm.add_constant(X)                              # Adds a constant term to linear equation
                                                    # If not added, it will assume y = mx

model = sm.OLS(Y, X).fit()                          # OLS regression with the best fit line
print(model.summary())                              # Provides a summary of the OLS output