## Using matplotlib and pandas library
## Note - the code won't work because the path doesn't have the resp. file

# Creating Pie charts

import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_excel('Course 1 Module 4.xlsx', sheet_name = 'Pie Chart')
labels = data['Sector']
frequencies = data['Freq']
plt.pie(frequencies, labels = labels, autopct = '%1.1f%%')
plt.show()

# Creating Tree maps

import matplotlib.pyplot as plt
import pandas as pd
import squarify as sqr
data = pd.read_excel('Course 1 Module 4.xlsx', sheet_name = 'Tree Map')
labels = data['Sector']
frequencies = data['Freq']
sqr.plot(sizes = frequencies, label = labels)
plt.show()