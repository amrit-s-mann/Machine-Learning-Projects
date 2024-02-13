''' This lab builds on the case study: Clustering Stocks Based on Co-Movement Similarity. 
It will expand on your understanding of how unsupervised machine learning algorithms such as 
1) AGGLOMERATIVE CLUSTERING 
2) K-MEANS CLUSTERING
3) DIVISIVE CLUSTERING
are used for addressing investment problems involving clustering of stocks'''


''' Importing all relevant libraries ''' 

import pandas as pd
import plotly
import plotly.figure_factory as ff
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import copy
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

init_notebook_mode(connected=True)  

''' Step 1: Collect panel data on adjusted closing prices for the stocks under investigation. '''

# The 8 S&P 500 member stocks
names=['JPM', 'UBS', 'GS', 'META', 'AAPL', 'GOOG', 'GM', 'F']

# Load data
SP500=pd.DataFrame()

# Use a for loop to load different files into single dataframe
for name in names:
    df=pd.read_csv( name + '.csv', index_col='Date')
    SP500[name]=df['Adj Close']

# Round the number value to keep two decimals
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Log dataframe information
SP500.head()
SP500.tail()

# Using graph_objects
stock = 'AAPL'

import plotly.graph_objects as go

SP500R = SP500.reset_index()
fig = go.Figure([go.Scatter(x=SP500R['Date'], y=SP500R[stock])])
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text= stock)
# fig.show(renderer="iframe") # for running with local tools
fig.show() # for running inline in browser environment

''' Step 2: Calculate daily returns for each stock '''

# Transfer data to percentage of change
SP500_pct_change = SP500.pct_change().dropna()

# Round the number value to keep three decimals
pd.set_option('display.float_format', lambda x: '%.3f' % x)
SP500_pct_change.head()

# Using graph_objects
fig = px.line(SP500_pct_change, x=SP500_pct_change.index, y=SP500_pct_change.columns, title="Stock Daily Return")
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Daily Return")
fig.show()

import plotly.express as px

SP500_pct_change = SP500_pct_change.rename_axis(index='date', columns='company')
fig = px.area(SP500_pct_change, facet_col='company', facet_col_wrap=2)

# Define a function to update the y-axis title based on the position of the y-axis
def update_yaxis_title(yaxis_obj):
    yaxis_obj.title.text = 'Daily Return'
    
# Use the `for_each_yaxis` method to apply the function to each y-axis
fig.for_each_yaxis(update_yaxis_title,col=1)
fig.show()

''' Step 3: Distance matrix computation '''

from scipy.spatial import distance

# Init empty dataframe as a two dimension array
SP500_distances = pd.DataFrame(index=names, columns = names, dtype=float)

# Use two for loop to calculate the distance
for sym1 in names:
    for sym2 in names:
            SP500_distances[sym1][sym2] = distance.euclidean(SP500_pct_change[sym1].values,
                                                             SP500_pct_change[sym2].values)

# Explore the result
import seaborn as sns

fig = plt.figure(figsize=(14, 10))
sns.heatmap(SP500_distances, annot = True, fmt='.3f', vmin=0, vmax=0.5, cmap= 'coolwarm', xticklabels=True, yticklabels=True)
fig.show()

''' Performing AGGLOMERATIVE CLUSTERING analysis with visualization '''

## The **Dendrogram** is a convenient way of visualizing hierarchichal clusters. 
## Below we define and create a dendrogram using `scipy` library. 
## Vertical distance connecting various clusters represents euclidean distance between clusters. 
## Linkage is performed by averaging the distances

color_threshold =  0.36

# Draw figure using scipy and get data in function return as dendro
plt.figure(figsize=(16, 6))
plt.xlabel("Symbol")
plt.ylabel("Dissimilarity")
dendro = dendrogram(linkage(SP500_pct_change.T.values, method = 'average', metric = 'euclidean'), labels=names, color_threshold=color_threshold)

# Draw figure again with threshold showing
plt.figure(figsize=(16, 6))
plt.xlabel("Symbol")
plt.ylabel("Dissimilarity")
dendro = dendrogram(linkage(SP500_pct_change.T.values, method = 'average', metric = 'euclidean'), labels=names, color_threshold=color_threshold)

# Cutting the dendrogram at threshold
plt.axhline(y=color_threshold, c='k', linestyle='dashdot')

# Explore data
for i in dendro:
  print(i,dendro[i])

  # Generate clustering result by color using code
color_map = {}
leaves_cluster = [None] * len(dendro['leaves'])
for link_x, link_y, link_color in zip(dendro['icoord'],dendro['dcoord'],dendro['color_list']):
  for (xi, yi) in zip(link_x, link_y):
    if yi == 0.0:  # if yi is 0.0, the point is a leaf
      # xi of leaves are      5, 15, 25, 35, ... (see `iv_ticks`)
      # index of leaves are   0,  1,  2,  3, ... as below
      leaf_index = (int(xi) - 5) // 10
      # each leaf has a same color of its link.
      if link_color not in color_map:
        color_map[link_color] = len(color_map)
      leaves_cluster[leaf_index] = color_map[link_color]
leaves_cluster
# Or by observation directly
# leaves_cluster = [2, 0, 0, 1, 1, 1, 1, 1]

# Generate a dataframe storing the comparison result table
df_cluster = pd.DataFrame(leaves_cluster, columns=['Agglomerative'])
df_cluster.index = dendro['ivl']
df_cluster.sort_index(inplace=True)
df_cluster

# Generate a dictionary storing the comparison results
def decode_clusters(labels, clusters):
  result = {}
  for i in range(len(clusters)):
    if clusters[i] not in result:
      result[clusters[i]] = []
    result[clusters[i]].append(labels[i])
  return list(result.values())
result_comparison = {}
result_comparison['Agglomerative'] = decode_clusters(dendro['ivl'], leaves_cluster)
result_comparison

''' Performing K-MEANS CLUSTERING analysis with comparison table '''

import numpy as np
from sklearn import cluster

# Perform K-means using 'cluster' library and define the relevant hyperparameters
cl = cluster.KMeans(init='k-means++', n_clusters=3, max_iter=10000, n_init=1000, tol=0.000001)
cl.fit(np.transpose(SP500_pct_change))
cl.labels_

# Update the dataframe storing the comparison result table
df_cluster['K-means']=df_cluster['Agglomerative']
df_cluster['K-means'][SP500_pct_change.columns]=cl.labels_
df_cluster.sort_index(inplace=True)
df_cluster

# Update the dictionary storing comparison results
result_comparison['K-means'] = decode_clusters(SP500_pct_change.columns, cl.labels_)
result_comparison

''' Performing DIVISIVE CLUSTERING analysis with comparison table '''

# Define the hyperparameter
num_clusters = 3 


# Performing divisive clustering
import numpy as np;
import pandas as pd

all_elements = copy.copy(names)
dissimilarity_matrix = pd.DataFrame(SP500_distances,index=SP500_distances.columns, columns=SP500_distances.columns)

def avg_dissim_within_group_element(ele, element_list):
  max_diameter = -np.inf
  sum_dissm = 0
  for i in element_list:
    sum_dissm += dissimilarity_matrix[ele][i]   
    if( dissimilarity_matrix[ele][i]  > max_diameter):
      max_diameter = dissimilarity_matrix[ele][i]
  if(len(element_list)>1):
    avg = sum_dissm/(len(element_list)-1)
  else: 
    avg = 0
  return avg

def avg_dissim_across_group_element(ele, main_list, splinter_list):
  if len(splinter_list) == 0:
    return 0
  sum_dissm = 0
  for j in splinter_list:
    sum_dissm = sum_dissm + dissimilarity_matrix[ele][j]
  avg = sum_dissm/(len(splinter_list))
  return avg
    
def splinter(main_list, splinter_group):
  most_dissm_object_value = -np.inf
  most_dissm_object_index = None
  for ele in main_list:
    x = avg_dissim_within_group_element(ele, main_list)
    y = avg_dissim_across_group_element(ele, main_list, splinter_group)
    diff= x -y
    if diff > most_dissm_object_value:
      most_dissm_object_value = diff
      most_dissm_object_index = ele
  if(most_dissm_object_value>0):
    return  (most_dissm_object_index, 1)
  else:
    return (-1, -1)
    
def split(element_list):
  main_list = element_list
  splinter_group = []    
  (most_dissm_object_index,flag) = splinter(main_list, splinter_group)
  while(flag > 0):
    main_list.remove(most_dissm_object_index)
    splinter_group.append(most_dissm_object_index)
    (most_dissm_object_index,flag) = splinter(element_list, splinter_group)
  return (main_list, splinter_group)

def max_diameter(cluster_list):
  max_diameter_cluster_index = None
  max_diameter_cluster_value = -np.inf
  index = 0
  for element_list in cluster_list:
    for i in element_list:
      for j in element_list:
        if dissimilarity_matrix[i][j]  > max_diameter_cluster_value:
          max_diameter_cluster_value = dissimilarity_matrix[i][j]
          max_diameter_cluster_index = index
    index +=1
  if(max_diameter_cluster_value <= 0):
    return -1
  
  return max_diameter_cluster_index

current_clusters = ([all_elements])
level = 1
index = 0
result = None
while(index!=-1):
  if (result is None) and (len(current_clusters) >= num_clusters):
    result = copy.deepcopy(current_clusters)
    print(level, '*', current_clusters)
  else:
    print(level, current_clusters)
  (a_clstr, b_clstr) = split(current_clusters[index])
  del current_clusters[index]
  current_clusters.append(a_clstr)
  current_clusters.append(b_clstr)
  index = max_diameter(current_clusters)
  level +=1

if result is None:
  result = current_clusters
  print(level, '*', current_clusters)
else:
  print(level, current_clusters)

# Update the dataframe storing the comparison result table
df_cluster['Divisive'] = df_cluster['Agglomerative']
for i in range(len(result)):
  for col in result[i]:
    df_cluster['Divisive'][col]=i
    
## Or by observation directly, we get " df_cluster['Divisive'] = [2, 0, 0, 0, 0, 0, 1, 1] " ##
df_cluster.sort_index(inplace=True)
df_cluster

# Update the dictionary storing comparison results 
result_comparison['Divisive'] = result
result_comparison
