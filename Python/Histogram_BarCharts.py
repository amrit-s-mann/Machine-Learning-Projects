## Using matplotlib and pandas library
## Note - the code won't work because the path doesn't have the resp. file

# Creating Histogram

import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_excel('Course 1 Module 4.xlsx',sheet_name = 'Histogram')
plt.hist(data['Index Return'], bins = 20)
plt.show()

# Creating bar chart with axis

import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_excel('Course 1 Module 4.xlsx', sheet_name = 'Bar Chart')
sector = data['Sector']
frequency = data['Freq']
plt.bar(sector, frequency)
plt.show()

# Creating bar chart with axis and bars with different colour

import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_excel('Course 1 Module 4.xlsx', sheet_name = 'Bar Chart')
sector = data['Sector']
frequency = data['Freq']
plt.bar(sector, frequency, color = ['red', 'blue', 'green', 'yellow', 'black', 'maroon'])
plt.show()