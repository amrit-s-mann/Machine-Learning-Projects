## Using Statistics library 

# Arithmetic mean

import statistics
data1 = [3.5,-1.2,2.1,-1.4,2.6]
data2 = [6.3,1.4,4.2,3.1,-1.5]
data3 = [3.3,-0.9,2.4,1.6,2.2]

x = statistics.mean(data1)
y = statistics.mean(data2)
z = statistics.mean(data3)

print(x, y, z)

# Mean, mode, median

import statistics
data3=[6.1,-1.5,3.5,6.2,3.0,-1.0,-1.2,3.5,3.1,-0.9, 1.2]

x = statistics.mean(data3)
y = statistics.median(data3)
z = statistics.mode(data3)

print(x, y, z)

# Harmonic mean

import statistics
data=[22.29,15.54,9.38,15.12,10.72,14.57,7.2,7.97,10.34,8.35]
x=statistics.harmonic_mean(data)
print(round(x,2))

## Using numpy library

# Arithmetic mean

import numpy as np
data = [6.3,1.4,4.2,3.1,-1.5]
x = np.mean(data)
print(x)

# Calculating Median 

import numpy as np
data2=[-5.4,6.3,1.2,-3.1,-3.0,2.3,10.2,5.2,1.2,2.2,3.4]
data3=[6.1,-1.5,3.5,6.2,3.0,-1.0,-1.2,3.5,3.1,-0.9, 1.2]

x=statistics.median(data2)
y=statistics.median(data3)
print(x,y)

# Calculating weighted mean

from numpy import average
distribution=(1.2,6.7,3.4)
weight=(0.25,0.45,0.3)
weighted_avg_m3=round(average(distribution,weights=weight),2)
print(weighted_avg_m3)