''' Import Packages needed to run '''
## Note - Tensorflow library is not available on pip install for python 3.12 ##

import pandas as pd
import numpy as np
import re, math, random
import timeit, os, csv
import statsmodels.api as sm
import datetime as dt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l1,l2
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from dateutil.relativedelta import *

# Set default plotting method
%matplotlib inline

#Load Tensorboard extension
%load_ext tensorboard


''' Step 1: Define Callbacks '''

# Use this to kill existing instances if need be

from tensorboard import notebook

notebook.list() # View open TensorBoard instances
#!kill 1349

#Callbacks
es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=3)

logdir = os.path.join("logs", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0, write_graph=True, write_images=False)

''' Step 2: Define helper functions '''

# Define neural network model in Keras
def NN(n_inputs, n_units = 10, dropout =0.1, l1_reg =0.001, activation='relu', L=2):  
    model = Sequential()
    model.add(Dense(units=n_units, input_dim=n_inputs, kernel_regularizer=l1(l1_reg), kernel_initializer='normal', activation=activation))
    model.add(Dropout(dropout))
    for i in range (0,L-1):
        model.add(Dense(units=n_units, kernel_regularizer=l1(l1_reg), kernel_initializer='normal', activation=activation))
        model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer='normal')) 
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
    return(model)

# NN parameter tuning:
def parameter_tuning(X, y, cv=3, n_epochs=100, n_batch=10, seed = 7):
   param_grid = dict(n_inputs=[X.shape[1]],n_units=[10,20,50], l1_reg = [0, 0.0001, 0.001], activation=['relu','tanh']) # dropout=[0, 0.1, 0.2, 0.3],  #n_hidden_neurons=[10,50,100], 
   estimator = KerasRegressor(build_fn=NN, epochs=n_epochs, batch_size=n_batch, verbose=0)   
   grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv,fit_params=dict(callbacks=[es]))
   grid_result = grid.fit(X, y)
 
   print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
   means = grid_result.cv_results_['mean_test_score']
   stds = grid_result.cv_results_['std_test_score']
   params = grid_result.cv_results_['params']
   for mean, stdev, param in zip(means, stds, params):
         print("%f (%f) with: %r" % (mean, stdev, param))

# Training
def training(X, Y, training_dates, L, tune=False):
  n_epochs = 100  # maximum number of epochs (to be used with early stopping)
  n_batch = 10     # mini-batch size
  drop_out = 0.0   # level of dropout (set between 0 and 1)
  n_units = 10
  l1_reg = 0.001   # L_1 regularization parameter
  activation='tanh'
  models = {}
  xs = {}
  ys = {}
  
  models = {}
  models['linear']=[]
  models['NN']=[]
  xs = {}
  ys = {}

  xs['train']=[]
  xs['test']=[]
  ys['train']=[]
  ys['test']=[]
  
  betas = {}
  betas['NN'] =[]
  betas['linear'] =[]

  i=0
  for date in training_dates:
    start_time = timeit.default_timer()  
    print(i, date)
    train_index = X[X['date']==date].index
    
    if len(train_index)==0:
      continue
    date_next=pd.Timestamp(date).to_pydatetime() + relativedelta(months=+1)
    date_next = date_next.strftime('%-m/%-d/%Y')
    test_index = X[X['date']==date_next].index
    if len(test_index)==0:
      continue
    
    x_train = X.loc[train_index]
    x_train=x_train.drop("date", axis=1)
    y_train = Y.loc[train_index]
    y_train= y_train.drop("date", axis=1)
    x_test  = X.loc[test_index]
    x_test=x_test.drop("date", axis=1)
    y_test =  Y.loc[test_index]
    y_test=y_test.drop("date", axis=1)
    
    n_inputs = x_train.shape[1]
    if n_inputs ==0:
      continue
  
    if tune: # just perform parameter tuning once
      print("Parameter tuning with X-validation...")      
      parameter_tuning(x_train, y_train, 3)
      tune=False

    model = NN(n_units=n_units, n_inputs=n_inputs, dropout=drop_out, l1_reg=l1_reg, activation=activation, L=L)
    model.fit(x_train.values, y_train.values, epochs=n_epochs, batch_size=n_batch, verbose=0, callbacks=[es,tensorboard_callback])   
    beta=sensitivities(model, x_train.values, L, activation)   
    models['NN'].append(model)
    betas['NN'].append(beta)
    x=sm.add_constant(x_train)
    model =sm.OLS(y_train, x).fit()   
    betas['linear'].append(model.params)
    models['linear'].append(model)
    xs['train'].append(x_train)
    xs['test'].append(x_test)
    ys['train'].append(y_train)
    ys['test'].append(y_test)

    elapsed = timeit.default_timer() - start_time
    print("Elapsed time:" + str(elapsed) + " (s)")
    i+=1
      
  return models, betas, xs, ys, i

# Assume that the activation function is tanh
def sensitivities(lm, X, L, activation='tanh'):
    W=lm.get_weights()
    M = np.shape(X)[0]
    p = np.shape(X)[1]
    beta=np.array([0]*M*(p+1), dtype='float32').reshape(M,p+1)
    B_0 =W[1]
    for i in range (0,L):
      if activation=='tanh':  
        B_0 = (np.dot(np.transpose(W[2*(i+1)]),np.tanh(B_0))+W[2*(i+1)+1])
      elif activation=='relu':
        B_0 = (np.dot(np.transpose(W[2*(i+1)]),np.maximum(B_0,0))+W[2*(i+1)+1])
    
          
    beta[:,0]= B_0 # intercept \beta_0= F_{W,b}(0)
    for i in range(M):
      I1 = np.dot(np.transpose(W[0]),np.transpose(X[i,])) + W[1]
      if activation=='tanh':
          Z= np.tanh(I1)  
          D = np.diag(1-Z**2)
      elif activation=='relu':
          Z=np.maximum(I1,0)
          D = np.diag(np.sign(Z)) 
               
      for j in range(p):
        J = np.dot(D,W[0][j])       
        for a in range (1,L):
          I= np.dot(np.transpose(W[2*a]),Z) + W[2*a+1] 
          if activation=='tanh':  
              Z = np.tanh(I)
              D = np.diag(1-Z**2)
          elif activation=='relu':    
              Z=np.maximum(I,0)
              D = np.diag(np.sign(Z)) 
          J = np.dot(np.dot(D,np.transpose(W[a*2])),J)
        beta[i,j+1]=np.dot(np.transpose(W[2*L]),J)
            
    return(beta)

''' Step 3: Load data '''

# Load the data
X=pd.read_csv('X_small.csv')
Y=pd.read_csv('Y_small.csv')

X.head()
Y.head()

''' Step 4: Training of models '''

# Clear any logs from previous runs
import shutil
shutil.rmtree('logs/', True)

training_periods = 100
L = 2 
dates = np.unique(X['date'])[0:training_periods]
models, betas, xs, ys, training_periods = training(X,Y,dates,L,False) # set last argument to True to perform cross-validation for parameter tuning

''' 
Step 5: Visualize results

If you are new to Tensorboard, please feel free to refer to this [tutorial]
(https://www.tensorflow.org/tensorboard/get_started).

'''

# Display inline in notebook (if running locally)
#%tensorboard --logdir "./logs" --port 6006

# Display inline in lab notebook (port and host required for lab environment)
%tensorboard --logdir "./logs" --host 0.0.0.0 --port 8000 --reuse_port True

# Use this to kill existing instances if need be
from tensorboard import notebook

# View open TensorBoard instances
notebook.list() 
#!kill 1514

''' Step 6: Performance evaluation '''

def compute_MSE(model_type,data_type,xs,ys,training_periods):
  MSE = 0
  y_hat = []
  MSE_array=np.array([0]*training_periods,dtype='float64')
  for i in range(training_periods):
    if(model_type=='linear'):
      x= sm.add_constant(xs[data_type][i].values)
    else:
      x = xs[data_type][i].values
    y_hat.append(models[model_type][i].predict(x))
    MSE_array[i]= mean_squared_error(y_hat[-1], ys[data_type][i].values)
    MSE+=MSE_array[i]
  print("MSE:" + str(MSE/training_periods))  
  return MSE_array 

# Evaluate MSE on OLS out-of-sample
MSE_array_linear = compute_MSE('linear','test',xs,ys,training_periods)

# Evaluate MSE of OLS in-sample
MSE_array_linear_in = compute_MSE('linear','train',xs,ys,training_periods)

# Evaluate MSE of NN out-of-sample
MSE_array_NN = compute_MSE('NN','test',xs,ys,training_periods)

# Evaluate MSE of NN in-sample
MSE_array_NN_in = compute_MSE('NN','train',xs,ys,training_periods)

''' Step 7: Plotting '''

# Helper function for plotting
def plot_Line_Chart(y_NN,y_OLS,title):
  x = np.arange(len(y_NN))
  # Create traces
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x, y=y_NN,
                      mode='lines',
                      name='NN'))
  fig.add_trace(go.Scatter(x=x, y=y_OLS,
                      mode='lines',
                      name='OLS'))

  fig.update_layout(title=title,
                    xaxis_title='Period t',
                    yaxis_title='MSE')

  # fig.show(renderer="iframe") # for running with local tools
  fig.show() # for running inline in browser environment

  plot_Line_Chart(MSE_array_NN_in,MSE_array_linear_in,'MSE values in-sample')

  plot_Line_Chart(MSE_array_NN,MSE_array_linear,'MSE values out of sample')

  ''' Step 8: Sensitivities '''

  # Compute and plot sensitivities
n_var=np.shape(betas['NN'][1])[1]-1
mu = np.array([0]*training_periods*n_var, dtype='float32').reshape(training_periods,n_var)
sd = np.array([0]*training_periods*n_var, dtype='float32').reshape(training_periods,n_var)
mu_ols = np.array([0]*training_periods*n_var, dtype='float32').reshape(training_periods,n_var)

for i in range(training_periods):
 mu[i,:]=np.median(betas['NN'][i], axis=0)[1:]
 sd[i,:]=np.std(betas['NN'][i],axis=0) [1:]
 mu_ols[i,:]=betas['linear'][i][1:]
 
names = ['EV', 'P/B', 'EV/T12M EBITDA', 'P/S' , 'P/E','Log CAP']

fig, axes = plt.subplots(2, 1, figsize=(20, 10))
fig.subplots_adjust(hspace=.5)
fig.subplots_adjust(wspace=.7)
#%sc=10000
idx=np.argsort(np.median(mu[:,0:n_var], axis=0))
axes[0].boxplot(mu[:,idx])       # make your boxplot
axes[0].set_xticklabels(np.array(names)[idx],rotation=45)  
axes[0].set_ylim([-0.1,0.1])
axes[0].set_ylabel('Sensitivity (NN)')
axes[0].set_xlabel('Factor')
idx=np.argsort(np.median(mu_ols[:,0:n_var], axis=0))
axes[1].boxplot(mu_ols[:,idx]) 
axes[1].set_ylim([-0.1,0.1])
axes[1].set_xticklabels(np.array(names)[idx],rotation=45) 

axes[1].set_ylabel('Sensitivity (OLS)')
axes[1].set_xlabel('Factor')

## The above sensitivities are sorted in ascending order from left to right. 
## We observe that the OLS regression is much more sensitive to the factors than the NN.