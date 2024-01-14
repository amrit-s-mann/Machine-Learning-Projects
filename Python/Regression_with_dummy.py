## Import the libraries first

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm                                        # Note the use of .api
from sklearn.linear_model import LinearRegression

## Load the data

df = pd.read_excel("Data_science_course/Data/Helper_Data.xlsx", sheet_name = 'Dummies')
print(df.columns)
print(df)

## Visualising the data

plt.scatter(df.Period, df.Volatility,marker='o',color = 'red')
plt.xlabel('Year month')
plt.ylabel('Realised volatility')
plt.xticks(np.arange(0, len(df.Period)+1, 12), rotation = 45)       # Note that rotation is part of xticks, not arange
plt.show()

## Creating dummies and remove extras

pd.get_dummies(df, prefix=['FDIDummy'], columns=['Reforms'])
df=pd.get_dummies(df, prefix=['FDIDummy'], columns=['Reforms'],drop_first=True)
print(df.columns)
print(df)

## Regression with / without Dummies

# Without dummies
X = df.Period
Y = df.Volatility
X = sm.add_constant(X)
olsresult_nodummy = sm.OLS(Y, X).fit()
print(olsresult_nodummy.summary())

# With dummies
X=df['FDIDummy_Pre-reforms']
Y=df.Volatility
X = sm.add_constant(X)
olsresult_dummy = sm.OLS(Y, X).fit()
print(olsresult_dummy.summary())