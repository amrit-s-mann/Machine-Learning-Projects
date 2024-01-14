## Using matplotlib and pandas library
## Note - the code won't work because the path doesn't have the resp. file

# Creating scatter plot from excel file

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel('Course 1 Module 4.xlsx', sheet_name = 'Scatter Plot')
x = df['P/E']
y = df['Alpha']

plt.scatter(x, y, marker = '.')
plt.title('P/E vs Alpha')
plt.xlabel('P/E multiple')
plt.ylabel('Alpha')
plt.show()

# Creating scatter plot from raw data

import matplotlib.pyplot as plt
x=[106,108,109,113,115,120,121,132,125,126]
y=[2.0,2.1,2.1,2.3,2.4,2.6,2.6,2.7,2.8,3.0]
plt.scatter(x,y)
plt.xlabel('Bond Yield')
plt.ylabel('Stock Price')
plt.show()

