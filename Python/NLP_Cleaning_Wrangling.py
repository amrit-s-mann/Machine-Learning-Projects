''' This sample program is part one of four coding labs. 
The four coding labs demonstrate NLP techniques for classifying data. 
In this module, the focus is on cleaning and wrangling the input data
'''

''' PART 1 - Importing libraries and files '''

import numpy as np
import pandas as pd

import string
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from itertools import islice
#Plotly Tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

#We will need "punkt" from the NLTK library for tokenizing
nltk.download('punkt')

'''The NLTK library will be required; better to pip install nltk
#nltk.download()
#nltk.download('punkt')
'''

# Change the path and make sure input files are present in this directory
path ='C:\\Users\\sxk\\Downloads\\SVK 2 programs\\DSC4\\'
filesentences = 'DSC 4 Module 1 Sentences.txt'

''' PART 2 - Reading input data into Python data frames '''

''' 2A Reading the input data from a flat file '''

# Reading the data, the data was stored using latin1 encoding, the separator between fields is "@" and there are no headers,
# so use these parameters

useencoding = 'latin1'
dfdata = pd.read_csv(path+filesentences,header=None,encoding=useencoding,sep='@',names=['sentence','sentiment'])

#The shape command tells us that there are 4840 rows and 2 columns
dfdata.shape

# Get a feel for the data by checking the contents of the dataframe
dfdata

''' 2B Removing "neutral" sentiment sentences. Getting numerical values for categories '''

# We want to demonstrate two class classification so get rid of the neutrals
# We also need to replace the text values negative and positive with 0 and 1 respectively

sentiments = ['positive','negative']
dfdata = dfdata[(dfdata.sentiment.isin(sentiments))]

# Convert categorical data into a numerical format. Creating a new column nsentiment
dfdata['nsentiment'] = pd.factorize(dfdata['sentiment'], sort=True)[0]

''' PART 3 - Checking Values '''

# Check values in the newly created nsentiment column
dfdata['sentiment'].unique()
dfdata['nsentiment'].unique()
dfdata.loc[:,['nsentiment','sentiment']]

''' PART 4 - Cleaning '''

''' 4A Checking if a particular data set is being loaded correctly '''

#The raw data had a row with this string "Jan. 6 -- Ford".. just making sure that the string is there
dfdata.loc[dfdata.sentence.str.contains('Jan. 6 -- Ford') == True,:]

''' 4B Picked up location 30 ( a random location) to check what the "sentence" looks like '''

#Check value for the sentence column in any row. I chose row 31 
dfdata.iloc[30,0]

#Let us see how this value changes

''' 4C The real cleaning '''

# This code involves 16 data cleaning transformations.

# For removing punctuations
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

#1 For removing leading and trailing white spaces
dfdata['sentence'] = dfdata['sentence'].str.strip()

#2 For removing extra white space
dfdata['sentence'] = dfdata['sentence'].str.split().apply(' '.join)

#3 For changing % to percentsign (replace symbols to simple text)
dfdata['sentence'] = dfdata['sentence'].str.replace('%',' percentSign ',regex = False)

#4 For changing % to dollarsign (replace symbols to simple text)
dfdata['sentence'] = dfdata['sentence'].str.replace('$',' dollarSign ',regex = False)

#5 For changing apostropge to a blank 
dfdata['sentence'] = dfdata['sentence'].str.replace("'", "")

#6 For changing changing mn and mln to million
dfdata['sentence'] = dfdata['sentence'].str.replace(' mn ',' million ')

dfdata['sentence'] = dfdata['sentence'].str.replace(' mln ',' million ')

#7 For 1234 mn Or 1234 mln or 1234 m Or 1234m Or 1234mln..
strpattern = r'[\d]+[ ]*m+[ ln]+'
strsubstitute = ' million '
dfdata = dfdata.replace(to_replace =strpattern, value = strsubstitute, regex = True)

#8 Repeat for billions. For 1234 bn Or 1234 bln or 1234 b Or 1234b Or 1234bln..
strpattern = r'[\d]+[ ]*b+[ ln]+'
strsubstitute = ' billion '
dfdata['sentence'] = dfdata['sentence'].replace(to_replace =strpattern, value = strsubstitute, regex = True)

#9 For replacing " b " with a billion (need to know what the data contains)
dfdata['sentence'] = dfdata['sentence'].str.replace(' b ',' billion ')

#10 For replacing " illion" with a space
dfdata['sentence'] = dfdata['sentence'].str.replace(' illion',' ')

#11 For removing all punctuations (fullstops, commas etc)
dfdata['sentence'] = dfdata['sentence'].apply(remove_punctuations)

#12 For removing all digits and replacing these with space
strpattern = r'[\d]+'
strsubstitute = ' '
dfdata['sentence'] = dfdata['sentence'].replace(to_replace =strpattern, value = strsubstitute, regex = True)

#13 Another way to remove extra white space - regex
strpattern = r' +'
strsubstitute = ' '
dfdata['sentence'] = dfdata['sentence'].replace(to_replace =strpattern, value = strsubstitute, regex = True)

#14 Replace illion - this may have gotten left out
dfdata['sentence'] = dfdata['sentence'].str.replace(' illion', ' ')

#15 Replace the 's ' (as a result of replacing apostrophe - i.e 's)
dfdata['sentence'] = dfdata['sentence'].str.replace(' s ', 's ')

#16 Normalizing currency signs to currencies
dfdata['sentence'] = dfdata['sentence'].str.replace(' eur ', ' currencysign ')
dfdata['sentence'] = dfdata['sentence'].str.replace(' euro ', ' currencysign ')
dfdata['sentence'] = dfdata['sentence'].str.replace(' EUR ', ' currencysign ')
dfdata['sentence'] = dfdata['sentence'].str.replace(' EURO ', ' currencysign ')
dfdata['sentence'] = dfdata['sentence'].str.replace('EUR', ' currencysign ')

dfdata['sentence'] = dfdata['sentence'].str.replace(' usd ', ' currencysign ')
dfdata['sentence'] = dfdata['sentence'].str.replace(' pence ', ' currencysign ')
dfdata['sentence'] = dfdata['sentence'].str.replace('eur ', ' currencysign ')
dfdata['sentence'] = dfdata['sentence'].str.replace(' eur', ' currencysign ')

# Perhaps these variants are present as well

#dfdata['sentence'] = dfdata['sentence'].str.replace('eur ', ' currencysign ')
#dfdata['sentence'] = dfdata['sentence'].str.replace(' eur', ' currencysign ')
#dfdata['sentence'] = dfdata['sentence'].str.replace('eur', ' currencysign ')

''' 4D Check how location 30 in 4b has changed '''

#Lets check how the value in row 31 has changed (30 refers to 31 as the counting begins at 0)
#Note how the "o" in "euro" was overlooked as the code just replaced "eur". Replace the "euro" phrase as well

dfdata.iloc[30,0]

''' PART 5 - Wrangling
Wrangling involves defining tokens, creating a bag of words, normalizing the tokens in the bag of words,
creating a document term matrix. This would mean transforming the text data so it can be expressed by
numbers in rows and columns. sklearn offers libraries that take care of both wrangling and sometimes
cleaning too. The illustration here is just to showcase the steps involved.
'''

''' 5A Creating a bag of words. As part of the normalizing process, one essential step after cleaning 
is to get to the root of words. This is done through stemming '''

#Preprocessing

#Check - how many tokens?
ps = PorterStemmer()
dfdata['tokenized'] = dfdata['sentence'].apply(word_tokenize)
print(dfdata['tokenized'].apply(len).sum())
# Returns 39686 

# Get stems (note we are getting stems from the original "sentence" column)
dfdata['stemsentence'] = dfdata['sentence'].apply(ps.stem)

# Remove extra white space for stemmed sentence - regex
strpattern = r' +'
strsubstitute = ' '

dfdata['stemsentence'] = dfdata['stemsentence'].replace(to_replace =strpattern, value = strsubstitute, regex = True)

dfdata['stemtokenized'] = dfdata['stemsentence'].apply(word_tokenize)

print(dfdata['stemtokenized'].apply(len).sum())
# Returns 39685

# For getting word counts
dfdata['wordcount'] = dfdata.stemsentence.str.replace(',','').str.split().str.len()

# The column stemtokenzied is now a bag of words - 1 gram features
# functions to create n gram token are offered as part of functionality 

''' 5B We are using a sklearn library to get the document term matrix '''

# This creates/ sets the model up. Each of the parameters have a significance.
# cvector = CountVectorizer(stop_words='english', min_df=1, max_df=.5, ngram_range=(1,1)) 

cvector = CountVectorizer(stop_words='english') 

ctdm = cvector.fit_transform(dfdata.stemsentence)
# This is the vocabulary
#print(cvector.vocabulary_)
# These are the features - check them out. Try changing to bi-gram by changing the ngram_range parameter
print(cvector.get_feature_names())
#print(ctdm.toarray())

# This is the term document matrix
dfbow = pd.DataFrame(ctdm.toarray(), columns=cvector.get_feature_names())
print(dfbow)

# Merge the original df with the term document matrix dataframe to see how each sentence fares 
dfbow = dfbow.reindex(dfdata.index)
dfalltdm = pd.concat([dfdata, dfbow], axis=1)
print(dfalltdm)

#Check location 31 iloc - just check the original sentence and the first 10 columns, note that the number of features are more than 2000 - the bag of words in case of a unigram
print('Location 30')
print(dfalltdm.iloc[30,0:9])
