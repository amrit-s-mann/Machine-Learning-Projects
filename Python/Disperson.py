## Using statistics library

import statistics
data = [3.5, -1.2, 2.1, -1.4, 2.6, 1.8, 2.4, 3.1]
x = statistics.variance(data)
y = statistics.stdev(data)
print(round(x, 2),round(y, 2))

import statistics
data2=[6.3,1.4,4.2,3.1,-1.3,3.5,2.2,2.4]
data3=[3.3,-0.9,2.4,1.6,2.2,-1.2,1.8,2.0]
X2=statistics.variance(data2)
X3=statistics.variance(data3)
Y2=statistics.stdev(data2)
Y3=statistics.stdev(data3)
print(round(X2,2),round(Y2,2),round(X3,2),round(Y3,2))