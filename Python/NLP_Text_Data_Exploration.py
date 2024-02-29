''' This sample program is part two of four coding labs. 
The four coding labs demonstrate NLP techniques for classifying data. 
In this module, the focus is on exploratory data analysis
'''

''' PART 1 - Importing libraries and files '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# The NLTK library will be required; better to pip install nltk
#nltk.download()
#nltk.download('punkt')

# Change the path and make sure input files are present in this directory

path = '' #'C:\\Users\\sxk\\Downloads\\SVK 2 programs\\DSC4\\'
filesentences = 'DSC 4 Module 2 Sentences.txt'

''' PART 2 - Reading input data into Python data frames '''

# Reading the data, the data was stored using latin1 encoding, the separator between fields is "@" 
# and there are no headers, so use these parameters

useencoding = 'latin1'
dfdata = pd.read_csv(path+filesentences,header=None,encoding=useencoding,sep='@',names=['sentence','sentiment'])

#The shape command tells us that there are 4840 rows and 2 columns
dfdata.shape

# Get a feel for the data by checking the contents of the dataframe
dfdata

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

''' PART 4 - Cleaning
Cleaning data comprises a number of important steps. These steps could change from dataset to dataset. 
But in general these steps involve (among many others) removing double spaces and leading and trailing 
spaces, replacing signage such as % and or the $ sign for extracting "meaning"
'''

#The raw data had a row with this string "Jan. 6 -- Ford".. just making sure that the string is there

dfdata.loc[dfdata.sentence.str.contains('Jan. 6 -- Ford') == True,:]

#Check value for the sentence column in any row. I chose row 31 
dfdata.iloc[30,0]
#Let us see how this value changes

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

#Lets check how the value in row 31 has changed (30 refers to 31 as the counting begins at 0)
dfdata.iloc[30,0]
#Note how the "o" in "euro" was overlooked as the code just replaced "eur". 
#Replace the "euro" phrase as well

''' PART 5 - Exploratory data analysis
EDA involves looking analyzing and understanding the data set in the context of the outcomes being 
sought. This will require looking at the sample data, finding out features that matter and looking 
for outlier events
'''

''' 5a. Take a look at the raw data (before that do the wrangling too) '''

# What does the data really look like?

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
dfdata['stemsentence']

''' 5b. What are the top words in this data set? How frequently do these occur? '''

# Getting top unigrams before removing stop words 

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(dfdata['stemsentence'], 20)
for word, freq in common_words:
    print(word, freq)
dfcwf = pd.DataFrame(common_words, columns = ['sentence' , 'count'])
dfcwf.groupby('sentence').sum()['count'].sort_values(ascending=False).iplot(
kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in sentences before removing stop words')

''' 5c What happens if we remove the stop words? '''

# Getting top unigrams after removing stop words 

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(dfdata['stemsentence'], 20)
for word, freq in common_words:
    print(word, freq)
dfcwf = pd.DataFrame(common_words, columns = ['sentence' , 'count'])
dfcwf.groupby('sentence').sum()['count'].sort_values(ascending=False).iplot(
kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in sentences after removing stop words')

''' 5d Working with bigrams - before stopwords are removed '''

# Getting top bigrams before removing stop words 

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(dfdata['stemsentence'], 20)
for word, freq in common_words:
    print(word, freq)
dfcwf = pd.DataFrame(common_words, columns = ['sentence' , 'count'])
dfcwf.groupby('sentence').sum()['count'].sort_values(ascending=False).iplot(
kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in sentences before removing stop words')

''' 5e Working with bigrams - after stopwords are removed '''

# Getting top bigrams after removing stop words 

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english',ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(dfdata['stemsentence'], 20)
for word, freq in common_words:
    print(word, freq)
dfcwf = pd.DataFrame(common_words, columns = ['sentence' , 'count'])
dfcwf.groupby('sentence').sum()['count'].sort_values(ascending=False).iplot(
kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in sentences before removing stop words')

'''  5f Check, recheck, also take a look at location 30 (row 31)! '''

# Just making sure that we dont have a "," in the data
#dfdata.sentence.str.len()
dfdata.stemsentence.str.replace(',','').str.split().str.len()
dfdata.stemsentence

'''
Check what has happened to location 31
Column 0 contains the original, cleaned sentence
Column 3 contains tokenized values
Column 5 contains stemmed and tokenized values

Let us see how this value changes (I am using tolist so that details in the column are not limited to display lengths)
Observe how the original sentence has changed from being a sentence, to a bunch of tokens, to a bunch of stemmed tokens
The difference between the "bunch of tokens" column and "stemmed bunch of tokens" is not much
'''

dfdata.iloc[30,[0,3,5]].to_list()

''' PART 6 - Visualization'''

''' 6a Using WordCloud '''

# Get positive sentiment words
pcloudtext = dfdata.loc[dfdata.nsentiment == 1, ['stemsentence']].values
pwordcloud = WordCloud(stopwords=STOPWORDS, width = 800, height = 800,background_color ='white',min_font_size = 10).generate(str(pcloudtext))

# Get negative sentiment words
ncloudtext = dfdata.loc[dfdata.nsentiment == 0, ['stemsentence']].values
nwordcloud = WordCloud(stopwords=STOPWORDS,width = 800, height = 800,background_color ='black',min_font_size = 10).generate(str(ncloudtext))

# Wordcloud for positive sentiment sentences
plt.imshow(pwordcloud)
plt.figsize=(250,250)
plt.title("Positive Sentiment Cloud", fontsize=15)
plt.axis("off")
plt.show()

# Wordcloud for negative sentiment sentences
plt.imshow(nwordcloud)
plt.figsize=(250,250)
plt.title("Negative Sentiment Cloud", fontsize=15)
plt.axis("off")
plt.show()

''' 6b Checking random values - features for positive and negative sentiment respectively '''

print('5 random values with a positive sentiment: \n')
dfcheck = dfdata.loc[dfdata.nsentiment == 1, ['stemtokenized']].sample(5).values
for checkvalue in dfcheck:
    print(checkvalue[0])
    
print('5 random values with a negative sentiment: \n')
dfcheck = dfdata.loc[dfdata.nsentiment == 0, ['stemtokenized']].sample(5).values
for checkvalue in dfcheck:
    print(checkvalue[0])

''' 6c Using sklearn for extracting features & creating a term document matrix '''

# This creates/ sets the model up. Each of the parameters have a significance.
# cvector = CountVectorizer(stop_words='english', min_df=1, max_df=.5, ngram_range=(1,1)) 

cvector = CountVectorizer(stop_words='english') 

ctdm = cvector.fit_transform(dfdata.stemsentence)
# This is the vocabulary
# print(cvector.vocabulary_)
# These are the features - check them out. Try changing to bi-gram by changing the ngram_range parameter
print(cvector.get_feature_names())
# print(ctdm.toarray())
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

''' 6d Using transformer to show IDF weights '''

# Transforms the count matrix we created earlier to a normalized tf or tf-idf representation.
transformer = TfidfTransformer()
cvectortransform = cvector.transform(dfdata.stemsentence)
transformed_df = transformer.fit_transform(cvectortransform)

idfweights = np.asarray(transformed_df.mean(axis=0)).ravel().tolist()
idfweights_df = pd.DataFrame({'term': cvector.get_feature_names_out(), 'idfweight': idfweights})
idfweights_df.sort_values(by='idfweight', ascending=False).head(20)


'''
SOME ADDITIONAL USEFUL EXPERIMENTS

The code you have contains a function that will return the “n” most frequently occurring words. 
Currently, the function is returning the top 20 most frequently occurring words.

Can you modify this function to return the top 30 words (after removing stop words)? 
How does the bar graph change? (Hint: Change the call “common_words = get_top_n_words” from 20 to 30.)

Try using the “get_top_n_words” function to determine the 30 least frequently occurring words? 
This can help identify rare words that may be important. 
(Hint: Change the parameter in the “words_freq = sorted” call from reverse=True to reverse=False.)

'''