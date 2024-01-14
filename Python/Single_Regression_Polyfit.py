# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load excel file
db = pd.read_excel("Data_science_course/Data/Inflation.xlsx", sheet_name = 'Inflation')

#Check content
print(db)

# Assign X and Y from columns in the dataframe
X = db.Money_Supply                                 # Column with values of independent variable
Y = db.Inflation                                    # Column with values of dependent variable
slope, constant = np.polyfit(X, Y, 1)               # Regression function in numpy library
print("Slope: " + str(round(slope, 3)))             # Note usage of str function to print value
print("Constant: " + str(round(constant, 3)))      # Note usage of str function to print value
plt.plot(X, Y, 'o')                                 # Plots the scatter of input & output
plt.plot(X, slope*X + constant, 'red')              # Plots the regression fit line
plt.xlabel('Money Supply %')
plt.ylabel('Inflation %')
plt.show()
