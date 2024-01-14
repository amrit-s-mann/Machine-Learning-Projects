# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load excel file
path = 'Data_science_course/Data/'
df = pd.read_excel(path + 'For ANOVA.xlsx', sheet_name = 'ANOVA')

# Check content
print(df)

# Assign X and Y from columns in the dataframe
X = df[['Interest_Rate', 'Unemployment_Rate']]      # Column with values of independent variable
Y = df['Index_Price']                               # Column with values of dependent variable

X = sm.add_constant(X)                              # Adds a constant term to linear equation
                                                    # If not added, it assumes y = m1x1 + m2x2

ols_result = sm.OLS(Y, X).fit()                     # OLS regression with the best fit line
print(ols_result.summary())                         # Provides a summary of the OLS output