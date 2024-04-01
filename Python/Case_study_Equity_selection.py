'''---------------------------------------------------------------------------------------
CASE STUDY OBJECTIVE

This use case develops three equities selection models in a global equities universe. 
- **Traditional quant:** First, we examine the risk of biases of traditional quant models, based on a linear factor approach, identifying problematic assumptions and violations.
- **Hybrid Machine Learning/quant:** Second, we "enhance" the traditional approach using machine learning (ML): traditional quant+ML. We will see that this presents additional risks and biases, as we attempt to improve a traditional quant model using a neural net.
- **Interpretable machine learning equities selection**: Third, we examine a purpose designed ML approach, designed to address many of the possible biases in both traditional quant and traditional quant+ML. 

------------------------------------------------------------------------------------------ '''

''' Code Repositories '''

# package for working with tabular data
import pandas as pd 
import numpy as np

# package for navigating the operating system
import os

# Progress bar
from tqdm.notebook import tqdm

# Pretty dataframe printing for this notebook
from IPython.display import Markdown, display

# Suppress warnings for demonstration purposes...
import warnings

# Type checking
import numbers

''' Github Repo

Many of the functions we use in this notebook are wrapped up in functional classes on the in the Investment-SAI repository, and some in the FinGov repository. For students with coding background and the interest, we encourage you to review these classes and functions.
'''

# install the key packages...
'''
!pip install scipy==1.9.3 -q
!pip install investsai -q
!pip install InvestmentToolkit -q
'''

# Dependency: We may have to install SHAP, an explainable AI (XAI) package onto your machine (or your Google Colab session if you are running this notebook in Colab)
# !pip install shap -q
import shap

# imblearn is required for FinGov utilities
# !pip install imblearn -q

# Clone the FinGov repo which has key utility functions for this notebook
# if os.path.exists('FinGov') == False:
  #  !git clone https://github.com/cfa-pl/FinGov

# Now import the objects from the repo
os.chdir('FinGov')
print('Current working directory: ' + os.getcwd())

print('Importing objects from FinGov repo...')
from GovernanceUtils import GovernanceUtils
from FairnessUtils import FairnessUtils
import CFACreditUseCaseUtils

# Set CWD back to the notebook directory
os.chdir('..')
print('Current working directory: ' + os.getcwd())

# Install SAI and other packages we will need

# Import classes of the InvestmentToolkit which has key utility functions we will use in this notebook
from InvestmentToolkit.itk import SimulationUtils
from InvestmentToolkit.itk import RobustInvestmentUtils
from InvestmentToolkit.itk import LinearFactorInvesting
from InvestmentToolkit.itk import NonLinearFactorInvesting
from InvestmentToolkit.itk import SAIInvesting

'''---------------------------------------------------------------------------------------
Stage 1: Business Case

We need to define a ground truth, then hypotheses and then design an experimental setup to test them. In this notebook our ground truth is simply securities that generate higher returns. Note that this is prone to bias, as we may have more objectives than just outright return delivery.
------------------------------------------------------------------------------------------ '''

'''---------------------------------------------------------------------------------------
Stage 2: Data

We will use a subset of a US small cap equities universe, defined as US equities with a market cap between aproximately $150m$ and $3bn$, removing equities with incomplete data. Note that by removing equities we may expose ourselves to survivorship bias. We will use 10 years of fundamental and pricing data.
First we pull in the data we need, wrangle it, examine it and prepare it.
------------------------------------------------------------------------------------------ '''

''' Stage 2a: Load Data '''

# Get returns and fundamentals for our universe and load into DataFrames

# Step into the data directory...
os.chdir('data')
print('Current working directory: ' + os.getcwd())

# Extract security level returns - Monthly frequency
df_raw_sec_rets = pd.read_csv('TR_US_SMID.csv')
# Extract security level fundamentals - Monthly frequency
df_raw_sec_ff = pd.read_csv('FF_US_SMID.csv', encoding = "ISO-8859-1")

df_raw_sec_rets

''' 
Selecting a subset of securities for this use-case
We select a subset of securities for this use-case to reduce processing time, and to provide a good illustration of the processes we are demonstrating. Note that this selection procedure is itself exposed to biases. 
'''

# confirm the current working directory
print('Current working directory: ' + os.getcwd())

# load the subset of securities to use as an example...
# Get the subset of equities
df_full_tr_hist_sec_rets = pd.read_csv('securities_subset.csv', index_col='TICKER')

# print the equity universe we will be using...
print('Equities Universe for this Notebook')
display(df_full_tr_hist_sec_rets.index)

# reduce data
secs = df_full_tr_hist_sec_rets.index.unique()

df_raw_sec_rets = df_raw_sec_rets[(df_raw_sec_rets['TICKER'].isin(secs) == True)]
df_raw_sec_ff = df_raw_sec_ff[(df_raw_sec_ff['TICKER'].isin(secs) == True)]

# Extract returns for 5 US reference portfolios, or factors, and the risk free rate (RF, the 3month T-Bill yield) from the Ken French data library.

# confirm the current working directory
print('Current working directory: ' + os.getcwd())

# Libraries...
import urllib.request
import zipfile
import csv

# Get factor returns from the Ken French data repository

# To download the original file yourself, run the following lines:
# ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
# urllib.request.urlretrieve(ff_url,'fama_french.zip') # name it fama_french.zip

zip_file = zipfile.ZipFile('fama_french.zip', 'r')
# Next we extract the file data
zip_file.extractall()
# Make sure you close the file after extraction
zip_file.close()

# Extract into a dataframe
df_ff_factors = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', skiprows = 3)

# Data wrangling... remove invalid rows (annual factors)
# There is white space in the date columns... strip this out
df_ff_factors.iloc[:,0] = df_ff_factors.iloc[:,0].str.strip()

# Find the start of the invalid rows and remove them all...
for row in range(1,df_ff_factors.shape[0]):
  if df_ff_factors.iloc[row,0].isnumeric() == False:
    break
df_ff_factors = df_ff_factors.drop(df_ff_factors.index[row:])

df_ff_factors.index = df_ff_factors.iloc[:,0]
df_ff_factors = df_ff_factors.drop(columns=df_ff_factors.columns[0], axis=1)

df_ff_factors = df_ff_factors.sort_index(axis=0, ascending=False)
df_ff_factors

# Now that we have all the data files, set CWD back to the notebook directory
os.chdir('..')
print('Current working directory: ' + os.getcwd())

''' Stage 2b: Data Wrangling and Preprocessing

Shape the data (df_raw_sec_rets and df_raw_sec_ff) with dates as rows, securities as columns (and for df_ff_factors: dates as rows, factors as columns). 
The latest date should be the top row... dates should all be the same format in each file.
'''

# Returns and fundamental data: pivot data so that rows are dates (latest date in the top row), columns are factors or securities:

# Function to extract a single data item from the raw data. We will call it further below.
def extract_dataitem_from_ff(df_raw: pd.DataFrame,
                             col_to_extract: str = '') -> pd.DataFrame:
  
  # Sort columns and index
  df_extract = pd.pivot(df_raw, index='TICKER', columns='DATE', values=col_to_extract)
  df_extract = df_extract.sort_index(ascending=True)
  
  # Alter dates to YYYYMMDD format
  df_extract.columns = pd.DataFrame(df_extract.columns, index=None).iloc[:,0].apply(lambda x: int(x[6:] + x[0:2]))
  df_extract = df_extract.sort_index(ascending=False, axis=1)
  df_extract = df_extract.sort_index(ascending=False, axis=0)

  return df_extract

# Create a dictionary of all the fundamental/valuation/metric data items 
# Extract all fundamental/valuation/metric dataitems from raw data into a security/date matrix (ignore static data columns)

dict_sec_ff = dict()

#Extract all data items
for di in df_raw_sec_ff.columns:
  if di not in ['TICKER', 'SEC_NAME', 'ENT_NAME', 'COUNTRY_DESC', 'FACTSET_SECTOR_DESC', 'DATE','ADJDATE',	'ORIG_CURRENCY']:
    dict_sec_ff[di] = extract_dataitem_from_ff(df_raw_sec_ff, di)

# Backfill all values - so as we have financial reporting data populated between fiscal-end/filing dates
for idx in dict_sec_ff:
  if di not in ['TICKER', 'SEC_NAME', 'ENT_NAME', 'COUNTRY_DESC', 'FACTSET_SECTOR_DESC', 'DATE','ADJDATE',	'ORIG_CURRENCY']:
    for i in range(dict_sec_ff[idx].shape[1]):
      dict_sec_ff[idx].iloc[:, i:i+12] = dict_sec_ff[idx].iloc[:, i:i+12].fillna(method="bfill", axis=1)

''' 
Refining our universe
Remove equities with < 75% of returns populated; cap returns at +500% 
'''

# pivot total returns
df_sec_rets = extract_dataitem_from_ff(df_raw_sec_rets, 'MTD_RETURN_USD')
df_sec_rets = df_sec_rets/100

# Transpose
df_sec_rets = df_sec_rets.T

# ***********************************************
# Remove underpopulated **** Note that this could 
# Kill all securities with any na returns in the period.
secs_valid = (df_sec_rets.isna().sum(axis=0) < (df_sec_rets.shape[0] * 0.75))
df_sec_rets = df_sec_rets[df_sec_rets.columns[secs_valid]]
# ***********************************************

# Get the equities universe
secs_universe = list(dict_sec_ff.values())[0].index.unique()
secs_universe = [col for col in df_sec_rets.columns if col in secs_universe]

# Only use returns where we have valid fundamental data
df_sec_rets = df_sec_rets[secs_universe]

# Cap all returns to 500%
df_sec_rets[df_sec_rets>5] = 5

''' Prepare factor data '''

# Get FF factors sorted and transposed...
# Rows are dates (latest date in the top row), columns are factors or securities...
df_ff_factors = df_ff_factors.sort_index(ascending=False)
# Set row index as dates
df_ff_factors.index = df_ff_factors.index.astype(int)
# Force type to float
df_ff_factors = df_ff_factors.astype(dtype=float)
# Stated as percentages? No...
df_ff_factors = df_ff_factors/100

# Separate RF from the Factor DataFrame
df_tb3ms = df_ff_factors[['RF']]

# Set row index as dates
df_tb3ms.index = df_tb3ms.index.astype(int)

# Remove RF from the factor data
df_ff_factors = df_ff_factors[['Mkt-RF','SMB','HML','RMW','CMA']] # << AN error here may indicate we have picked up the wrong factor file.

# Let's check our data... rows are dates; equities/factors are columns; returns stated as percentages 

display(df_tb3ms.head())
display(df_ff_factors.head())
display(df_sec_rets.head())

# Mostly all good, but we have a date alignment issue. 

'''
Data Wrangling: Date alignment
Ensure date alignment across all data we are using... enforce a common end date by taking the earliest end date across our DataFrames. Enforce a  a common start date by taking the latest end date across our DataFrames.
'''

# Enforce the end date...
# Get the date_end that we will use, this will be the study end date...
date_end = min([max(df_tb3ms.index.astype(int)), max(df_ff_factors.index.astype(int)), max(df_sec_rets.index.astype(int))])

# Remove all date columns after the date_end
df_tb3ms = df_tb3ms.drop(index=df_tb3ms.index[df_tb3ms.index.astype(int) > date_end])
df_ff_factors = df_ff_factors.drop(index=df_ff_factors.index[df_ff_factors.index.astype(int) > date_end])
df_sec_rets = df_sec_rets.drop(index=df_sec_rets.index[df_sec_rets.index.astype(int) > date_end])

# Enforce the start date...
# Get the date_start that we will use, this will be the study start date...
date_start = max([min(df_tb3ms.index.astype(int)), min(df_ff_factors.index.astype(int)), min(df_sec_rets.index.astype(int))])

# Remove all date columns after the date_end
df_tb3ms = df_tb3ms.drop(index=df_tb3ms.index[df_tb3ms.index.astype(int) < date_start])
df_ff_factors = df_ff_factors.drop(index=df_ff_factors.index[df_ff_factors.index.astype(int) < date_start])
df_sec_rets = df_sec_rets.drop(index=df_sec_rets.index[df_sec_rets.index.astype(int) < date_start])

''' 
Sanity check the data for errors and mistakes. Eye ball the distributions and correlations too. Here are some of the checks that should be carried out at a minimum:

1. Date alignment errors
2. Check all percentages are formatted correctly
3. Check nan values have a low count
'''

''' 
Governance: Validation check on our data
Validate the input data to ensure it is correct, the dates are correctly aligned, formats and conventions are aligned (e.g., whether numbers are represented decimals (0.10) or percentages (10%)) and that there are no obvious issues with the data. 
'''

# Sanity checking
# 1: Dates are aligned?
if  (df_tb3ms.index.equals(df_ff_factors.index) == False) | (df_ff_factors.index.equals(df_sec_rets.index) == False):
  raise TypeError('Sanity: Dates are not aligned...')

# 2: Percentages are percentages (not decimals)? 
# Check df_tb3ms median is within median +/- 2standard deviations of df_ff_factors
med_abs = df_ff_factors.iloc[:,:].abs().median(skipna=True).median(skipna=True)
sd = df_ff_factors.iloc[:,:].std(skipna=True).median(skipna=True)
if (med_abs + sd*5 < df_tb3ms.iloc[0,:].abs().median(skipna=True)) | (med_abs - sd*5 > df_tb3ms.iloc[0,:].abs().median(skipna=True)):  
  raise TypeError('Sanity: df_ff_factors values to be outside of a sensible range...')  
if (med_abs + sd*5 < df_sec_rets.abs().median(skipna=True).median(skipna=True)) | (med_abs - sd*5 > df_sec_rets.abs().median(skipna=True).median(skipna=True)):  
  raise TypeError('Sanity: df_ff_factors values to be outside of a sensible range...')  

# Check df_ff_factors median is within median +/- 2standard deviations of df_ff_factors
med_abs = df_tb3ms.iloc[0,:].abs().median(skipna=True) # only one row... only one median call needed
sd = df_tb3ms.iloc[0,:].std(skipna=True) # only one row... no median call needed
if (med_abs + sd*5 < df_ff_factors.iloc[0,:].abs().median(skipna=True)) | (med_abs - sd*5 > df_ff_factors.iloc[0,:].abs().median(skipna=True)):  
  raise TypeError('Sanity: df_tb3ms values to be outside of a sensible range...')  
if (med_abs + sd*5 < df_sec_rets.abs().median(skipna=True).median(skipna=True)) | (med_abs - sd*5 > df_sec_rets.abs().median(skipna=True).median(skipna=True)):  
  raise TypeError('Sanity: df_tb3ms values to be outside of a sensible range...')  

# Check df_sec_rets median is within median +/- 2standard deviations of df_ff_factors
med_abs = df_sec_rets.abs().median(skipna=True).median(skipna=True)
# median of each row, median across rows
sd = df_sec_rets.std(skipna=True).median(skipna=True) # std of each row, median across rows
if (med_abs + sd*5 < df_tb3ms.iloc[0,:].abs().median(skipna=True)) | (med_abs - sd*5 > df_tb3ms.iloc[0,:].abs().median(skipna=True)):  
  raise TypeError('Sanity: df_sec_rets values appear to be outside of a sensible range...')  
if (med_abs + sd*5 < df_ff_factors.iloc[0,:].abs().median(skipna=True)) | (med_abs - sd*5 > df_ff_factors.iloc[0,:].abs().median(skipna=True)):  
  raise TypeError('Sanity: df_sec_rets values appear to be outside of a sensible range...')  

# 3: Many nan? 
if df_tb3ms[df_tb3ms==np.nan].count().sum() / (df_tb3ms.shape[0]*df_tb3ms.shape[1]) > 0.33:
  raise TypeError('Sanity: df_tb3ms; > 33% of values are nan')  
if df_ff_factors[df_ff_factors==np.nan].count().sum() / (df_ff_factors.shape[0]*df_ff_factors.shape[1]) > 0.33:
  raise TypeError('Sanity: df_ff_factors; > 33% of values are nan')  
if df_sec_rets[df_sec_rets==np.nan].count().sum() / (df_sec_rets.shape[0]*df_sec_rets.shape[1]) > 0.33:
  raise TypeError('Sanity: df_sec_rets; > 33% of values are nan')  

''' Stage 2c: Exploratory data analysis (EDA)

For investment models EDA should encompass exploring data types (interval, cardinal, nominal), distributions, correlations, and more. It is a good idea to examine pairwise distributions of input variables to ensure colinearities (and other relationships) do not exist.  
'''

# Functions for EDA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import shapiro

# Function we will call to add R2 and p-val to the off-diagonal cells of the pair plot
def R2func(x, y, hue=None, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    _, _, r, p, _ = stats.linregress(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate(f'p-val = {p:.2f}', xy=(.1, .8), xycoords=ax.transAxes)

# Function we will call to add normality test stat and p-val to diagonal cells of pair plot
# Note that inputs to linear regression are not required to be normally distributed.
def normalityfunc(x, hue=None, ax=None, **kws):
    """Plot the Shapiro Wilk p-value in the top left hand corner of diagonal cells."""
    stat, p = shapiro(x)
    ax = ax or plt.gca()
    ax.annotate(f'Shapiro-Wilk stat = {stat:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate(f'p-val = {p:.2f}', xy=(.1, .8), xycoords=ax.transAxes)

# Run EDA Functions on our dataset

# Generate pairplot
pp = sns.pairplot(df_ff_factors, kind='reg', diag_kind='kde',
             plot_kws={'line_kws':{'color':'red'}})

# Run the R2func for all off diagnonal cells, and normalityfunc for the diagonal ones...
pp.map_lower(R2func)
pp.map_upper(R2func)
pp.map_diag(normalityfunc)

# Title...
pp.fig.subplots_adjust(top=0.9) # adjust the Figure in rp

pp.fig.suptitle('Linear Regression Violation Check: Normality test for factors, pair-wise correlation test')

plt.show()

'''
Eyeballing our data we can see from the frequency distributions on the diagonal, that all variables have approximately normally distributed. It can also be noted that HML and CMA are somewhat correlated, which may cause colinearity issues with the linear regression models we will use later.
'''

'''---------------------------------------------------------------------------------------
Stage 3: Model Design
------------------------------------------------------------------------------------------ '''

'''
Bias Alert: Complexity Bias

There is a balancing act between model complexity and the data available, and if our model becomes too complex for the amount of data we will likely overfit to the data, and our model will not perform well out of sample. It will be a bad predictor. The more samples we have, and the more accurate the data, the more accurate our model will be. The more data we have the more features and model parameters  (ie complexity) we can employ to approximate our target function. 
In practice, historic simulations require relatively large amounts of data to avoid overfitting. A relatively long history of returns is required to learn a linear regrssion model with many input variables[1], less for a classification model. ML models generally have more parameters than traditional linear models and therefore we need even more data for training. In short, we may not have high enough frequency data to train a stable model. This problem is exaccerbated when running simulations, as to avoid data snooping biases we can only use a subset of our full dataset that preceeds each simulation date.   
'''

#***********************************************************************
#*** Complexity bias Sanity Check!***
#***********************************************************************  
def bias_complexity_check_regression(no_of_instances: int,
                          no_of_features: int = 0,
                          no_of_parameters: int = 0) -> (bool, int, int):

  '''
  Check the complexity of the mode based on rules of thumb.

  Args:
    no_of_instances: Number of rows in your dataset
    no_of_features: Number of columns
    no_of_parameters: Number of weights/coefficients/parameters in your model
      
  Returns:
      rf: sklearn model object
      
  Author:
      failed: Did the complexity check fail? Too complex...
      feature_max: maximum number of features you shluld have given the problem type and instances
      param_max: maximum number of weights/coefficients/parameters in your model given the problem type and instances
  '''

  failed = False
  cb_K = no_of_features
  cb_n = no_of_instances
  
  # 1. Feature complexity: n ≥ 50 K
  if cb_n > 50 * cb_K:
    failed = True
  
  feature_max = int(round(cb_n / 50, 0))

  # 2. Parameter complexity: ¦θ¦ ≤ n/10
  #
  # The number of model parameters (ie weights) should observe the constraint
  # wrt training instances, n, features, K:
  #
  # |theta| >= n/10  
  param_max = int(round(cb_n / 10, 0))
  
  if no_of_parameters > param_max:
    failed = True

  return (failed, feature_max, param_max)

'''----------------------------------------------------------------------------------------------
Stage 3a: Traditional Quantitative Approach, Linear Factors

We will test a simple, fundamental factor model applied to investing stocks in our universe, also known as a time-series factor model, after Fama and French. This has the form:
-------------------------------------------------------------------------------------------------

$$R_{it} = α_{i} + \sum\limits_{j=1}^{K} \beta_{ij} f_{ij} + ϵ_{ij} \qquad \text{where}\; i=1,\dots, N\; j=1,\dots, K$$

Where $R_{it}$ is the excess return of equity $i$ at time $t$, $\beta$ is the exposure (loading) of equity $i$, representing an observed source of inefficiency such as value, $f$ is the return of a reference portfolio $j$. 
In practice we will use a vectorized version of this factor model, which is much faster:

$$
R_{t} = \mathbf{α +β f_{t} + ϵ_{t}}
$$

We will first learn the loadings of each equity to each of the factors in the dataset, and we will then use these loadings and the assumed future factor returns to determine the expected return of each equity to form model portfolios.

----------------------------------------------------------------------------------------------

BIAS ALERT: Researcher bias

Biases that can arise due to the researcher being influenced in both data collection, data exploration and modelling approaches proposed by previous research. Factor investing, provides a number of possible areas for researcher bias, including model design and factors to use.

----------------------------------------------------------------------------------------------

BIAS ALERT: Sample Distribution Bias

If the training data and the actual data to be predicted come from different distributions, this would likely challenge to the generalization ability of a model (ie how well it performs on future input data). If a parametric model is used, such as a linear factor model, the risks of shifting distributions are likely to be greater still, as normality assumptions may be violated, even if normlization/standardization of the input data is carried out.

We now specify our model training process, using OLS regression. We will wrap this up as a function, so as we can call it multiple times to generate return simulations over the study term.

Note that we sanity check parameters, to provide basic checks, and we analyse the residuals produced by the model to ensure residuals are normally distributed (using a Shapiro-Wilk test in this case, with the option of generating a scatter plot in the function too).
Let us run the function, using our 5 factor regression on the first security in the dataset, column=0.

Many refinements can be made to this simple model.

----------------------------------------------------------------------------------------------

Stage 3a i) Prepare data

Our data is wrangled and ready but we need to shape it to pass it into our model to train and predict.

----------------------------------------------------------------------------------------------

Stage 3a ii) Equity Selection

As one function call we can train our model and predict our expected returns. Wrapped in this one function we train our model, passing in training data (in this case forming a training window of the past 36 months), ensuring that this data is at or before our assumed execution date, and    
'''

# Run the function with our data
df_all_er = LinearFactorInvesting.factormodel_forecast_all_er(df_benchmark_trades=None, # Only use df_benchmark_trades positions with a non NaN value
                                        df_tb3ms=df_tb3ms, # Risk free rate
                                        df_sec_rets=df_sec_rets, # security level returns histories, monthly in USD
                                        df_ff_factors=df_ff_factors, # factor returns monthly in USD
                                        window_size=36, # training window size
                                        factor_return_history=36) # prediction: return history of factors to assume will persist going forwards

# Test an example
test_this_equity = df_all_er.columns[0]
df_all_er.columns.get_loc(test_this_equity)

'''
Example: Expected Returns of a Single Equity 
Let us train a factor model for on a single equity in the dataset using the past 36months of data, and then calculate this security's expected returns. We can also check the residuals of this model and check over the statistics. Note that the residuals do not appear entirely normal.
'''

# Get an example equity
i = 0  
t = 19 # Run the model as if we were at a timepoint 19months ago.

with warnings.catch_warnings(): # Supress warnings
    warnings.simplefilter("ignore")

    # Run our function on the first security (sec_col_no=0), returning only the model object, and Shapiro-Wilk stat and p-value
    ols_model, y, y_hat = LinearFactorInvesting.factormodel_train_single_security(sec_col_no=i, 
                                                 df_tb3ms=df_tb3ms, 
                                                 df_sec_rets=df_sec_rets, 
                                                 df_ff_factors=df_ff_factors, 
                                                 date_start=t+36, 
                                                 date_end=t, 
                                                 plot_residual_scatter=True)
    # Examine the fit, coefficients and OLS analytics.
    print(ols_model.summary())

    # Forecast E(R) for this security.
    df_equity_er = LinearFactorInvesting.factormodel_forecast_er(df_stock_factor_loadings=pd.DataFrame(ols_model.params, index=None), 
                                    df_ff_factors=df_ff_factors, 
                                    r_f=df_tb3ms.iloc[0,0],
                                    date_start=t+36, 
                                    date_end=t,)

    print("Expected Return Forecast for this equity")
    print(df_equity_er)

''' Stage 3a iii) Construct Model Portfolios

In each period we select the top quartile of our equity forecasts, and generate a DataFrame of equal weighted equities that reside in this quantile.
Next we assume a 6month rebalance frequency, and simulate the past returns of this portfolio over the study term

----------------------------------------------------------------------------------------------

BIAS ALERT: Backtesting Bias

Back-testing bias includes forward-look (or data snooping) bias, where an analyst will check the performance of an approach over the past, and will tend to reject all approaches that perform poorly. In a sense the analyst becomes endogenous to the model, using a look-ahead to bias model selection to only those models that have worked well. 
This issue can be addressed by testing only a small number of approaches (i.e., avoiding violation of independence assumptions) with strong investment rationales (i.e., causality) and/or by comparing the simulated returns to the possible empirical outcomes using target shuffling. 

----------------------------------------------------------------------------------------------

BIAS ALERT: Time-Interval/Forward Look Bias

When researcher selects a specific timeframe for training vs. validating vs. testing to support hypotheses and/or uses “restated” data not available during the time under study. 
'''

# Get the top 25% of equities by exepected return over the study term
df_trades = SimulationUtils.trades_topquantile_generate(df_all_er=df_all_er, 
                                        rebalance_freq=6, 
                                        min_quantile_to_buy=0.75)

''' Stage 3a iv) Simulate Returns 

We can now conduct an historic simulation.  We iterate from the earliest time period where we have a model portfolio to the most recent period, generating expected returns for each equity in our model portfolio as we step forward through time, and simulating a rebalance event when there is a new model portfolio in a time period we step into. 
We combine step ii) Forecast Equity Level Expected Returns, iii) Select Top Quartile Equities, and iv) Simulate Returns, into the next code block:
'''

#====================================
# Run the Linear factor model
#====================================

# FF model
df_lin_all_er = LinearFactorInvesting.factormodel_forecast_all_er(df_benchmark_trades=None,
                                        df_tb3ms=df_tb3ms, 
                                        df_sec_rets=df_sec_rets,
                                        df_ff_factors=df_ff_factors,
                                        window_size=36,
                                        winsorize_er=0)

# Run the function to establish simple position sizes based
df_lin_trades = SimulationUtils.trades_topquantile_generate(df_all_er=df_lin_all_er, 
                                        rebalance_freq=6, 
                                        min_quantile_to_buy=0.75)

# Run the simulation function
df_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_lin_trades, 
                         rebalance_freq=6, 
                         df_sec_rets=df_sec_rets,
                         date_start_of_sim=79)  

# Now we construct a simple benchmark from our universe...

#====================================
# Create universe benchmark
#====================================
# Run the function to establish simple positions
df_benchmark_trades = SimulationUtils.trades_topquantile_generate(df_all_er=df_lin_all_er, rebalance_freq=6, min_quantile_to_buy=0)

df_sec_rets_copy = df_sec_rets.copy(deep=True)
df_sec_rets_copy[df_sec_rets_copy>5] = 5

# Create an equal weighted benchmark of all valid securities
# Run the simulation function
df_benchmark_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_benchmark_trades, 
                                    rebalance_freq=6, 
                                    df_sec_rets=df_sec_rets_copy, 
                                    date_start_of_sim=79)  

# Plot the linear-factor model CAGR vs the benchmark
df_lin_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_lin_trades, 
                         rebalance_freq=6, 
                         df_sec_rets=df_sec_rets,
                         print_chart=False,
                         date_start_of_sim=79)  

# Chart
p = SimulationUtils.sim_chart(df_sec_cagr)
SimulationUtils.sim_chart_add_series(p, df_benchmark_sec_cagr.sum(axis=1, skipna=True), 'Benchmark')
SimulationUtils.sim_chart_add_series(p, df_lin_sec_cagr.sum(axis=1, skipna=True), 'Linear factor Strategy', emphasize=True)
p.legend()
p.show()

''' Stage 3a v) Assess Returns

We can observe the total returns of our model and its performance against the benchmark but return simulations can give a misleading impression of how significant performance or risk adjusted returns would have been. As we saw in previous modules, we can construct an empirical distribution of possible returns using "target shuffling". This allows us to look at many of the combinations of securities we could have held over the study term which allows us to compare our results to the results of these randomly selected portfolios.  
'''

# get an empirical distribution of outcomes using target-shuffling "lite"
dt_target_shuffling_dist = RobustInvestmentUtils._target_shuffling_lite_get_dist(df_opportunity_set_trades=df_benchmark_trades, # Randomly allocate to equities in our universe
                                      min_quantile_to_buy = 0.75, # Same rebalance assumptions as our approach
                                      df_sec_rets=df_sec_rets, 
                                      rebalance_freq=6, # Same rebalance assumptions as our approach
                                      iterations=100) # Number of portfolios to randomly create and simulate

# check the performance of the using target shuffling lite, to form an empirical distributon of returns
display(Markdown('## **Factor Model Results**'))
RobustInvestmentUtils.target_shuffling_chart(dt_target_shuffling_dist, df_lin_sec_cagr)
# benchmark
display(Markdown('## **Benchmark Results**'))
RobustInvestmentUtils.target_shuffling_chart(dt_target_shuffling_dist, df_benchmark_sec_cagr)

# We can conclude that factor models would have done a poor job of adding value over the benchmark over this period of time, which may cause us to decide to reject this approach. 

'''
BIAS ALERT: Certainty bias

Significance tests do not by themselves provide a logically sound basis for concluding an effect is present or absent with certainty or a given probability. A broader analysis than just the hypothesis and the analyzed data (which give only statistical probabilities) must be used to reach such a conclusion.

----------------------------------------------------------------------------------------------

BIAS ALERT: Endogenous bias

Endogeneity bias is not a simple violation and there is no easy statistical solution. It has serious consequences for outcomes, where in the presence of endogenous variables OLS learns biased and inconsistent parameters. P-values can be seriously misleading. All it takes is one endogenous variable to seriously distort ALL OLS parameters.
One potentially concerning endogeneity is self-fulfilling prophecy of factor investing, where equities with high correlations to commonly used factors would cause investments in those equities, causing price appreciation, affirming that the correlations with those factors caused the price rises.

----------------------------------------------------------------------------------------------
Stage 3b: Factor Approach with Non-linear Regression

We now examine applying machine learning to a traditional quantative framework. Rather than linearly multiplying the expected returns of factors and stock level factor loadings as a simple linear factor approach does, we now train an MLP neural net to forecast expected returns based on stock level factor loadings and past factor returns.
----------------------------------------------------------------------------------------------

$$
R_{it} = \tilde{f}(\beta_{i,1}, ... , \beta_{i,K}, f_{1}, ..., f_{K})
$$

The MLP we will use has the following architecture, where the 11 input units relate to our 6, equity level factor coefficients (ie 5 $β$s, one $α$), and 5 corresponding factor returns over the longer run $\tilde{f}$. The architecture is shown with one hidden-layer, 3 hidden units, and one output $R$.

<div>
<img src="model.png" width="500"/>
</div>

----------------------------------------------------------------------------------------------

Stage 3b i) Prepare data 

We still need to calculate factor loadings at the stock level as before, but we now need to pass training data to our MLP. See the X (stock level factor loadings, assumed factor returns) and y variable (future return) we will be using below
'''

# Show the training data for training the MLP
X_nlf_train, y_nlf_train = NonLinearFactorInvesting.nonlinfactor_er_func_prep_data(df_tb3ms=df_tb3ms,
                               df_sec_rets=df_sec_rets,
                               df_ff_factors=df_ff_factors,
                               date_end=0,
                               func_training_period = 1)

print('X consists of stock level factor loadings, and assumed future factor returns')
display(X_nlf_train)
print('y consists of the stock level returns we want to forecast')
display(y_nlf_train)

''' Stage 3b ii) Equity Selection

Now we need functions to generate factor loadings at the stock level, then to train our neural net, and to provide the final forecast of equity returns in a given time period. We will need these function to generate expected returns.

Example: Expected Returns at a Single Time Point 
We can test our model, training it at a specific time point to observe the residuals of our model.
'''

# train nn 
nn_model, X_nlf, y_train_nlf, y_hat_nlf = NonLinearFactorInvesting.nonlinfactor_train_er_func(df_tb3ms=df_tb3ms, 
                                              df_sec_rets=df_sec_rets,
                                              df_ff_factors=df_ff_factors,
                                              date_end=0,
                                              forecast_ahead=6,
                                              window_size=36,
                                              func_training_period=12,
                                              plot_residual_scatter=True)

# Now we can implement a function that applies the forecasting function over the study term, ready for historic simulation testing.

# Run the function with our data
df_nlf_all_er, nn_mod_latest = NonLinearFactorInvesting.nonlinfactor_forecast_all_er(df_benchmark_trades=None, # which equities to include?
                                                            df_tb3ms=df_tb3ms, 
                                                            df_sec_rets=df_sec_rets,
                                                            df_ff_factors=df_ff_factors, # factor return data
                                                            window_size=36, # period to calculate equity loadings over
                                                            func_training_period=1) # how many periods should we stack up to train the MLP

''' 
BIAS ALERT: Complexity Bias

Using a complex model such as a neural net may be appealing but number of parameters the model needs to have trained may exceed our rule of thumb for complexity. It may in any case, not be an ideal algorthm to apply, if a more simple approach achieves a similar result. 
'''

# count the number of parameters in the MLP
param_count = 0
for i in range(0, nn_model.coefs_.__len__()):
  param_count += nn_model.coefs_[i].shape[0]


# Sanity Check: Biases ************************
failed, _, _ = RobustInvestmentUtils.bias_complexity_check_regression(no_of_instances=36, # Try to use  36 month window to train the MLP
                                    no_of_features=X_nlf.shape[1]-1, # Do not count intercept
                                    no_of_parameters=param_count) 
if failed == True:
  print("************ Complexity bias warning ***************")  
# Sanity Check: Biases ************************
  
''' Stage 3b iii) Construct Model Portfolios,  iv) Simulate Returns

Again, based on expected returns from the model, we construct model portfolios over the study term, and then run our historic simulation. We will also chart the returns of our benchmark and other simulation returns as a comparator. 
Note that the returns from the Non-Linear Factor approach will vary slightly each time they are run, owing to the stochastic nature of the neural net training. 
'''

# Run the function to establish simple trades
df_nlf_trades = SimulationUtils.trades_topquantile_generate(df_all_er=df_nlf_all_er, rebalance_freq=6, min_quantile_to_buy=0.75)

# Get the min common start date for the simulations (given the different training data windows they might use) so as we can compare them
start_date = min(int(SimulationUtils.start_period_trades(df_trades)),
                 int(SimulationUtils.start_period_trades(df_nlf_trades)),
                 int(SimulationUtils.start_period_trades(df_benchmark_trades)))

# Plot the non-linear-factor model CAGR vs the benchmark
df_non_lin_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_nlf_trades, 
                         rebalance_freq=6, 
                         df_sec_rets=df_sec_rets,
                         print_chart=False)  

# Run sims from the same start date.
print('Run simulations from a common start date...')
df_non_lin_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_nlf_trades, rebalance_freq=6, df_sec_rets=df_sec_rets, print_chart=False, date_start_of_sim=start_date)  
df_lin_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_lin_trades, rebalance_freq=6, df_sec_rets=df_sec_rets, print_chart=False, date_start_of_sim=start_date)  
df_benchmark_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_benchmark_trades, rebalance_freq=6, df_sec_rets=df_sec_rets, print_chart=False, date_start_of_sim=start_date)  

# Chart
print('Generate charts...')
p = SimulationUtils.sim_chart(df_non_lin_sec_cagr)
SimulationUtils.sim_chart_add_series(p, df_benchmark_sec_cagr.sum(axis=1, skipna=True), 'Benchmark')
SimulationUtils.sim_chart_add_series(p, df_lin_sec_cagr.sum(axis=1, skipna=True), 'Linear factor Strategy')
SimulationUtils.sim_chart_add_series(p, df_non_lin_sec_cagr.sum(axis=1, skipna=True), 'Non-Linear factor Strategy', emphasize=True)
p.legend()
p.show()

''' Stage 3b v) Assess Returns

First we need to understand how the NonLinear Factor Model is driving its returns, but we are using a black-box model which makes this difficult. To address this we use the SHAP explainable AI (XAI) package to examine how the input data is driving outputs of the model, using parameter importance.
'''

import shap

# This can take a while
# SHAP XAI on the training data
samples_to_use = 25
X_nlf_sample = shap.sample(X_nlf, samples_to_use)
explainer = shap.KernelExplainer(nn_model.predict, X_nlf_sample)

with warnings.catch_warnings(): # Supress warnings
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(shap.sample(X_nlf, samples_to_use))
    shap.summary_plot(shap_values, X_nlf_sample.values, feature_names=X_nlf_sample.columns)

'''
Now we examine whether the returns are unusually good (or bad) using the target shuffling lite approach. From the chart below we can see the model's performance versus the target shuffling distribution (vertical back line).

Note that this result will vary on each run of this notebook, as the train/test dataset splitting and the MLP training both have randomized elements.
'''

# Check the performance using target shuffling lite, to form an empirical distributon of returns
display(Markdown('## **Non-linear Factor Model**'))
RobustInvestmentUtils.target_shuffling_chart(dt_target_shuffling_dist, df_non_lin_sec_cagr)

'''----------------------------------------------------------------------------------------------
Stage 3c: Interpretable Machine Learning Approach 

An alternative to enhancing traditional quant investing with ML is to use a purpose designed ML approach. We will use Symbolic artificial intelligence (SAI) [1], an investment rules-learning ML approach which aims to avoid many of the biases in traditional quantitative investing, and potential biases in combining traditional quantitative investing with ML. We will test a simplified SAI approach now.
---------------------------------------------------------------------------------------------- '''

# Import the SAI package...

from investsai.sai import SAI 

''' Stage 3c. i) Prepare Data

We will use the same factor loadings data we used previously, except SAI can be trained on factor loadings fom many time periods (we use 6 period below: func_training_period=6). Our ground truth is stocks that generate a total return in the top 25% of the universe in 6 months time (forecast_ahead=6).
'''

# Get training and test data to pass into SAI 
sai_X, sai_y_class, sai_y_tr = SAIInvesting.sai_er_func_prep_data(df_tb3ms=df_tb3ms, 
                                  df_sec_rets=df_sec_rets,
                                  dic_fundamentals=None, #<< Pass populated dict None,
                                  df_ff_factors=df_ff_factors,
                                  date_end=0,
                                  buysell_threshold_quantile=0.333,
                                  forecast_ahead=6, # << Forecasting 6 months ahead
                                  window_size=36, # << Calculate factor loadings over this period
                                  func_training_period=6) ## << Use this many periods to train SAI
sai_X

'''
Note: In the SAI training data above we have stacked factor loadings for all the equities in our universe for period 6 to period 11 (note the "_6" suffix for the security tickers for data from period 6 for example). As we specified a "date_end=0", but a "forecast_ahead=6" SAI will use data from 6 periods ago to train the model with a y-variable from "date_end=0". This it to avoid data-snooping. We have 6 periods in the training data (6 to 11) owing to "func_training_period=6".
'''

''' Stage 3c ii) Forecast Equity Level Expected Returns '''

# Train our SAI model (factor loadings data only)
sai_mod, sai_X, sai_y, sai_y_hat = SAIInvesting.sai_train_er_func(df_tb3ms=df_tb3ms, 
                                  df_sec_rets=df_sec_rets,
                                  dic_fundamentals=None,
                                  df_ff_factors=df_ff_factors,
                                  date_end=0, #<< Training a model at the latest date
                                  buysell_threshold_quantile=0.333, #<< Predict stocks in the top 25% of return outcomes 
                                  lift_cut_off=1.0, #<< Only learn rules where the lift >125%
                                  forecast_ahead=6, #<< predict stocks returns in 6monmths time.
                                  window_size=36, #<< Factor loadings calculated over 36months
                                  func_training_period=12, #<< SAI training data will use 12x months of stacked loadings for the equity universe
                                  show_analytics=True)
sai_X[sai_X==0] = np.nan

'''
We can review the investment rules SAI has learned, where the first rule (mkt-rf_3, cma_3) indicate equities need to satisfy "mkt-rf_3" and "rmw_3": which means a market beta in the top third (tercile) ("_3" is the top, "_1" is the bottom tercile) combined with an RMW loading in the top third of the equity universe. 
'''

# Rules our model will be using, needing to show a reasonable "lift"
sai_mod.rules[(sai_mod.rules['causal_lift']>1.0)].reset_index(drop=True)

'''
We can now apply this same process across the full study term, generating rules in each period to drive expected return forecasts in each.

Warning: This cell might take a few minutes to run. It is expected to take. 5 - 6 minutes.
'''

# Run the function with our factor loadings data
df_sai_all_er, sai_mod_latest = SAIInvesting.sai_forecast_all_er(df_benchmark_trades=df_benchmark_trades, 
                                                    df_tb3ms=df_tb3ms, 
                                                    df_sec_rets=df_sec_rets,
                                                    dic_fundamentals=None,
                                                    df_ff_factors=df_ff_factors,
                                                    date_end=0,
                                                    buysell_threshold_quantile=0.333, #<< FLag the top and bottom stocks to make rules for
                                                    lift_cut_off=1.0, #<< Only learn rules where the lift > ...
                                                    forecast_ahead=6,
                                                    window_size=36,
                                                    func_training_period=6,
                                                    plot_residual_analytics=False) # << Make True to show the rules in every time period in the term


''' Stage 3c iii) Construct Model Portfolios, iv) Simulate Returns '''

# Run the function to establish simple trades
df_sai_trades = SimulationUtils.trades_topquantile_generate(df_all_er=df_sai_all_er, 
                                                            rebalance_freq=6, 
                                                            min_quantile_to_buy=0.75)

# Get the min common start date for the simulations (given the different training data windows they might use) so as we can compare them
start_date = min(int(SimulationUtils.start_period_trades(df_sai_trades)),
                 int(SimulationUtils.start_period_trades(df_trades)),
                 int(SimulationUtils.start_period_trades(df_nlf_trades)),
                 int(SimulationUtils.start_period_trades(df_benchmark_trades)))

# Plot the non-linear-factor model CAGR vs the benchmark
df_sai_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_sai_trades, 
                         rebalance_freq=6, 
                         df_sec_rets=df_sec_rets,
                         print_chart=False,
                         date_start_of_sim=start_date)  

# Run sims from the same start date.
print('Run simulations from a common start date...')
df_non_lin_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_nlf_trades, rebalance_freq=6, df_sec_rets=df_sec_rets, print_chart=False, date_start_of_sim=start_date)  
df_lin_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_lin_trades, rebalance_freq=6, df_sec_rets=df_sec_rets, print_chart=False, date_start_of_sim=start_date)  
df_benchmark_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_benchmark_trades, rebalance_freq=6, df_sec_rets=df_sec_rets, print_chart=False, date_start_of_sim=start_date)  

# Chart
print('Generate charts...')
p = SimulationUtils.sim_chart(df_sai_sec_cagr)
SimulationUtils.sim_chart_add_series(p, df_benchmark_sec_cagr.sum(axis=1, skipna=True), 'Benchmark')
SimulationUtils.sim_chart_add_series(p, df_lin_sec_cagr.sum(axis=1, skipna=True), 'Linear factor Strategy')
SimulationUtils.sim_chart_add_series(p, df_non_lin_sec_cagr.sum(axis=1, skipna=True), 'Non-Linear factor Strategy')
SimulationUtils.sim_chart_add_series(p, df_sai_sec_cagr.sum(axis=1, skipna=True), 'SAI Strategy', emphasize=True)
p.legend()
p.show()

'''----------------------------------------------------------------------------------------------
Stage 3c iv) Assess Returns

We now compare our model performance vs the empirical distribution generated by the target shuffling lite approach.
---------------------------------------------------------------------------------------------- '''

# Check the performance of the SAI simulation using target shuffling lite, to form an empirical distributon of returns
display(Markdown('## **SAI Model Results**'))
RobustInvestmentUtils.target_shuffling_chart(dt_target_shuffling_dist, df_sai_sec_cagr)

'''
BIAS ALERT: End point bias

End point bias refers to the biased selection of a given time period and end date we might use to assess a model's performance. It is notable that each approach we have tested has certain periods where it appears to be the "best", and assessing performance to the last elapsed period is to some extent arbitrary. We can deal with this to some extent by using different sub-time periods to assess model performance over, perhaps coinciding with different market conditions (analytics such as hit rate can also allow us to assess consistency of return delivery). We can also test our model out-of-sample, and in subsamples, such as in different countries and sectors to assess generalization.
'''

'''----------------------------------------------------------------------------------------------
Stage 3d: SAI Using Fundamental Data and Increasing Lift 

Having compared the SAI output to linear and non linear factor models, we can now train the SAI model on more than just the factor loadings, adding our unaltered fundamental data to the training also. We can also increase the lift our rules need to achieve to be used to 2.
---------------------------------------------------------------------------------------------- '''

# Get training and test data to pass into SAI 
sai_X, sai_y_class, sai_y_tr = SAIInvesting.sai_er_func_prep_data(df_tb3ms=df_tb3ms, 
                                  df_sec_rets=df_sec_rets,
                                  dic_fundamentals=dict_sec_ff, #<< Pass populated dict of raw fundamentals/valuations/ratios
                                  df_ff_factors=df_ff_factors,
                                  date_end=0,
                                  buysell_threshold_quantile=0.333,
                                  forecast_ahead=24,
                                  window_size=36,
                                  func_training_period=6)

display(sai_X)

'''
Note that we now have extra columns of data such as ff_net_mgn (net margin), ff_oper_mgn (operating margin), ff_roa (return on assets), ff_roe (return on equity), ff_rotc (return on total capital), ff_tcap (total capital), and ff_pbk (price to book). These were chosen arbitrarily and more columns of data can be added as appropriate.
'''

# Run the function with our factor loadings data
df_sai_all_features_all_er, sai_all_features_mod = SAIInvesting.sai_forecast_all_er(df_benchmark_trades=df_benchmark_trades, 
                                                    df_tb3ms=df_tb3ms, 
                                                    df_sec_rets=df_sec_rets,
                                                    dic_fundamentals=dict_sec_ff,
                                                    df_ff_factors=df_ff_factors,
                                                    date_end=0,
                                                    buysell_threshold_quantile=-0.333, #<< FLag the top and bottom stocks to make rules for
                                                    lift_cut_off=2.0, #<< Only learn rules where the lift >2 ...
                                                    forecast_ahead=6,
                                                    window_size=36,
                                                    func_training_period=6,
                                                    plot_residual_analytics=False)

''' Warning: This cell might take a few minutes to run. It is expected to take 5 - 6 minutes '''

# Run the function to establish simple trades
df_sai_all_features_trades = SimulationUtils.trades_topquantile_generate(df_all_er=df_sai_all_features_all_er, 
                                                                         rebalance_freq=6, 
                                                                         min_quantile_to_buy=0.75)

# Get the min common start date for the simulations (given the different training data windows they might use) so as we can compare them
start_date = min(int(SimulationUtils.start_period_trades(df_sai_all_features_trades)), 
                 int(SimulationUtils.start_period_trades(df_sai_trades)),
                 int(SimulationUtils.start_period_trades(df_trades)),#
                 int(SimulationUtils.start_period_trades(df_nlf_trades)),
                 int(SimulationUtils.start_period_trades(df_benchmark_trades)))

# Plot the non-linear-factor model CAGR vs the benchmark
df_sai_all_features_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_sai_all_features_trades, 
                                        rebalance_freq=6, 
                                        df_sec_rets=df_sec_rets,
                                        print_chart=False,
                                        date_start_of_sim=start_date)  

# Run sims from the same start date.
print('Run simulations from a common start date...')
df_sai_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_sai_trades, rebalance_freq=6, df_sec_rets=df_sec_rets, print_chart=False, date_start_of_sim=start_date)  
df_non_lin_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_nlf_trades, rebalance_freq=6, df_sec_rets=df_sec_rets, print_chart=False, date_start_of_sim=start_date)  
df_lin_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_lin_trades, rebalance_freq=6, df_sec_rets=df_sec_rets, print_chart=False, date_start_of_sim=start_date)  
df_benchmark_sec_cagr, p = SimulationUtils.run_sim(df_trades=df_benchmark_trades, rebalance_freq=6, df_sec_rets=df_sec_rets, print_chart=False, date_start_of_sim=start_date)  

# Chart
print('Generate charts...')
p = SimulationUtils.sim_chart(df_sai_all_features_sec_cagr)
SimulationUtils.sim_chart_add_series(p, df_benchmark_sec_cagr.sum(axis=1, skipna=True), 'Benchmark')
SimulationUtils.sim_chart_add_series(p, df_lin_sec_cagr.sum(axis=1, skipna=True), 'Linear factor Strategy')
SimulationUtils.sim_chart_add_series(p, df_non_lin_sec_cagr.sum(axis=1, skipna=True), 'Non-Linear factor Strategy')
SimulationUtils.sim_chart_add_series(p, df_sai_sec_cagr.sum(axis=1, skipna=True), 'SAI Strategy')
SimulationUtils.sim_chart_add_series(p, df_sai_all_features_sec_cagr.sum(axis=1, skipna=True), 'SAI EXTENDED FEATURES lift=1.2 Strategy', emphasize=True)
p.legend()
p.show()

# check the performance of the SAI All Features simulation using target shuffling lite, to form an empirical distributon of returns
display(Markdown('## **SAI All Features Model Results**'))
RobustInvestmentUtils.target_shuffling_chart(dt_target_shuffling_dist, df_sai_all_features_sec_cagr)

# We can directly examine the rules driving the latest SAI model 

# Rules our model will be using, needing to show a reasonable "lift"
sai_all_features_mod.rules[(sai_all_features_mod.rules['causal_lift']>2)].reset_index(drop=True)

'''----------------------------------------------------------------------------------------------
Stage 4: Model Deployment

Investment model deployment requires a rigorous change process, several levels of testing and sign off, asignment of responsibilities for the live operation of the process, models and data before deployiong the code to the cloud (or on native hardware). 
A key part of this stage is monitoring of data drift, and the monitoring of stakeholder KPIs which are principally for investors, and compliance needs.
For investors, the characteristics of the strategy that were marketed and agreed with the investor, should be monitored and reported regularly with sufficient executive oversight and repotring to support it.
For Compliance, regular checking that the investment elements of the strategy are fully compliant with the investment management agreement (IMA), which organisations will tend to have in place as a separate system in any case. 
----------------------------------------------------------------------------------------------'''

'''----------------------------------------------------------------------------------------------
Stage 5: Model Monitoring and Reporting
----------------------------------------------------------------------------------------------'''

''' Stage 5a: Data Drift

Our model is now in production and being used in practice. We need to monitor its stability. One approach is to monitor the distribution of the input data versus the data the model was trained with . If the distributions have shifted the model outcomes could be invalidated. Here we use our function data_drift_psi to monitor data drift.
Is it time for us to retrain our model?
'''

# Get Data
#================================  
sai_train_X, sai_train_y_class, _ = SAIInvesting.sai_er_func_prep_data(df_tb3ms=df_tb3ms, 
                                df_sec_rets=df_sec_rets,
                                dic_fundamentals=dict_sec_ff,
                                df_ff_factors=df_ff_factors,
                                date_end=0,  #<< Latest period
                                func_training_period=6,
                                buysell_threshold_quantile=0.333,
                                forecast_ahead=6, #<< Latest period
                                window_size=36) #<< Latest period

sai_test_X, sai_test_y_class, _ = SAIInvesting.sai_er_func_prep_data(df_tb3ms=df_tb3ms, 
                                df_sec_rets=df_sec_rets,
                                dic_fundamentals=dict_sec_ff,
                                df_ff_factors=df_ff_factors,
                                date_end=1,  #<< Latest period
                                func_training_period=1,
                                buysell_threshold_quantile=0.333,
                                forecast_ahead=0, #<< Latest period
                                window_size=36) #<< Latest period


# The test data may not contain all the columns in the train data. 
# we can add nan columns for the missing columns....
# Add blank column if a column exists in the training data and NOT in the test
missing_cols_to_add = [col for col in sai_test_X.columns if col not in sai_train_X.columns]
sai_train_X[[missing_cols_to_add]] = np.nan
missing_cols_to_add = [col for col in sai_train_X.columns if col not in sai_test_X.columns]
sai_test_X[[missing_cols_to_add]] = np.nan

# Prepare train and test data for data drift check
train_datadrift = sai_train_X # pd.concat([sai_train_X, sai_train_y], axis=1)
test_datadrift = sai_test_X # pd.concat([sai_test_X, sai_test_y], axis=1)

# Data drift check...
data_drift_features = GovernanceUtils.data_drift_psi(train_datadrift, test_datadrift, buckettype='bins',buckets=10,axis=1,single_variable=False)
data_drift_target = data_drift_features[-1]

# Print out the target PSI value:
print("Target PSI value is ",data_drift_target, "\n" )

''' Stage 5b: Challenger Models

We have tested three types of equity selection models in this notebook. We have selected the SAI approach, but we could still use the other models to provide another check on our live model, ie a **challenger model**.
'''


'''----------------------------------------------------------------------------------------------
AFTERWORD 

We have done our best to include the key themes of what we think are the critical stages of model development in this notebook, but clearly in practice many more checks and details would be added to each of the 5 model development stages to best ensure stakeholder's KPIs are met, and Governance standards would be as high as possible. Hopefully our example and key themes will provide insight to avoid many of the classic biases in model development.
----------------------------------------------------------------------------------------------'''
