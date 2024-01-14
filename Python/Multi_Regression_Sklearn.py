# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Load excel file
path = 'Data_science_course/Data/'
df = pd.read_excel(path + 'For ANOVA.xlsx', sheet_name = 'ANOVA')

# Check content
print(df)

# Assign variables from columns in the dataframe
X = df[['Interest_Rate', 'Unemployment_Rate']]      # Note the use of double square brackets
Y = df[['Index_Price']]                             # If single bracket used, gives KeyError

## NOT MANDATORY 
X_OLS = sm.add_constant(X)                          # New variable to avoid interference later
ols_result = sm.OLS(Y, X_OLS).fit()
print(ols_result.summary())
## Only added to obtain ANOVA results

Regressor = LinearRegression()                      # Defines a regression object
Regressor.fit(X, Y)                                 # Performs the regression using X & Y

print("Intercept: " + str(Regressor.intercept_))    # Prints the regression output
print("Coefficents: " + str(Regressor.coef_))

## Code for visualization ##

# Linspace creates even spacing for X-axis and Y-axis before meshgrid
# Meshgrid creates a mesh using linespace as inputs
x_surf, y_surf = np.meshgrid(np.linspace(X.Interest_Rate.min(), X.Interest_Rate.max(), 100),np.linspace(X.Unemployment_Rate.min(), X.Unemployment_Rate.max(), 100))

# Creating a dataframe for independent variables
Xvalues = pd.DataFrame({'Interest_Rate': x_surf.ravel(), 'Unemployment_Rate': y_surf.ravel()})

# Generating an array of predicted values of dependent variables from regression equation
fittedY=Regressor.predict(Xvalues)
fittedY=np.array(fittedY)

# Generating a figure of the plot
fig = plt.figure(figsize=(20, 10))                  # Sets the width and height of figure

# Modifying the plot and adding features like subplot, scatter and labels
ax = fig.add_subplot(111, projection='3d')
# Adds axis. 3 digits denotes position of subplot

ax.scatter(X['Interest_Rate'],X['Unemployment_Rate'],Y['Index_Price'],c='red', marker='o', alpha=0.5)
# alpha above meaning 0 for transparent and 1 for opaque

ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='blue', alpha=0.3)
# Creates a plot surface - describes a functional relationship between two independent variables
# and dependent variable, rather than showing the individual data points

# Adding the labels
ax.set_xlabel('Interest Rate')
ax.set_ylabel('Uneployment Rate')
ax.set_zlabel('Price Index')

# Final output
plt.show()