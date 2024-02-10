# Import ALL packages you need for different types of classifiers
# such as CART, SVM, KNN and Random Forest
# Note: Not all libraries will be required for each classifier

from sklearn import tree, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

'''
ydata_profiling is used for creating the Profil report 
# Currently no support for Python 3.12

from ydata_profiling import ProfileReport
'''

'''
Set default plotting method

%matplotlib inline
'''

# Load and format data
etf_features=['investment', 'size', 'net_assets', 'portfolio_stocks', 'portfolio_bonds', 'price_earnings','price_book','price_sales','price_cashflow', 'basic_materials', 'consumer_cyclical', 'financial_services', 'real_estate', 'consumer_defensive', 'healthcare', 'utilities', 'communication_services','energy','industrials', 'technology', 'fund_return_1month']
etf=pd.read_csv("ETFs_raw.csv", usecols=etf_features)
etf.loc[etf.portfolio_stocks==0, 'portfolio_stocks']=100 - etf.loc[etf.portfolio_stocks==0, 'portfolio_bonds']

# Subset data
sectors=['basic_materials', 'consumer_cyclical', 'financial_services', 'real_estate', 'consumer_defensive', 'healthcare', 'utilities', 'communication_services','energy','industrials', 'technology']
idx=np.where(np.sum(etf[sectors], axis=1)>0)
etf_clean=etf.loc[idx]
etf_clean.head()

# Label data based on one standard deviation of the mean
mean = np.mean(etf_clean['fund_return_1month'])
sigma = np.std(etf_clean['fund_return_1month'])
etf_clean['label']=0
etf_clean.loc[etf_clean['fund_return_1month']>=(mean+sigma), 'label']=1
etf_clean.loc[etf_clean['fund_return_1month']<=(mean-sigma), 'label']=-1
etf_clean.sample(5)

# Encoding categorical values
print(etf_clean['size'].unique())
etf_clean['cat_size'] =0
etf_clean.loc[etf_clean['size']=='Large','cat_size']=1
etf_clean.loc[etf_clean['size']=='Medium','cat_size']=0
etf_clean.loc[etf_clean['size']=='Small','cat_size']=-1
print(etf_clean['cat_size'].unique())

# Encoding categorical values
print(etf_clean['investment'].unique())
etf_clean['cat_investment']=0
etf_clean.loc[etf_clean['investment']=='Blend','cat_investment']=1
etf_clean.loc[etf_clean['investment']=='Growth','cat_investment']=0
etf_clean.loc[etf_clean['investment']=='Value','cat_investment']=-1
print(etf_clean['cat_investment'].unique())

# Define columns
etf_training_features = ['cat_investment', 'cat_size', 'net_assets', 'portfolio_stocks', 'portfolio_bonds', 'price_earnings','price_book','price_sales','price_cashflow', 'basic_materials', 'consumer_cyclical', 'financial_services', 'real_estate', 'consumer_defensive', 'healthcare', 'utilities', 'communication_services','energy','industrials', 'technology']
etf_training_features_w_label = etf_training_features  + ['label']

etf_clean

# Remove records with empty value
etf_clean_=etf_clean[etf_training_features_w_label].dropna()

# Snapshot the cleaned data
etf_clean_[etf_training_features_w_label].to_csv('ETF_clean.csv')

# Display & Explore cleaned data 
etf_clean_.head()
etf_clean_.describe()

'''
Display Pandas Profiling Report. Please note that loading may take 1-2 minutes

profile = ProfileReport(etf_clean_, title="Pandas Profiling Report", explorative=True)
profile.to_notebook_iframe()
'''

# Split into random train and test subsets
X = etf_clean_[etf_training_features]
y = etf_clean_['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

''' CART Modeling on ETF data '''

# Train the decision tree
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X_train, y_train)

# Visualize the tree
dot_data = tree.export_graphviz(clf, out_file=None,  
                filled=True, rounded=True,
                special_characters=True, feature_names=etf_training_features)
graph = graphviz.Source(dot_data)
from IPython.display import Image 
Image(graph.pipe(format="png"))

# Evaluate by Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
metrics.ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, values_format='d', cmap='Blues',ax=ax)
plt.show()

# Evaluate by scores
y_pred_CART=clf.predict(X_test)
accuracy_CART = metrics.accuracy_score(y_test, y_pred_CART)
f1_CART = metrics.f1_score(y_test, y_pred_CART, average='weighted')
df_compare_ETF = pd.DataFrame(columns=['Accuracy','F1'])
df_compare_ETF.loc['CART']={'Accuracy':accuracy_CART,'F1':f1_CART}
df_compare_ETF

