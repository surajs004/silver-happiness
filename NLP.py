#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Data handling and processing
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy import stats
import statsmodels.api as sm
from scipy.stats import randint as sp_randint
from time import time

# NLP
import nltk
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


# In[6]:


# Reading in data
data = pd.read_csv('C:/Users/Suraj S/Downloads/Womens Clothing E-Commerce Reviews.csv')
data = data[['Clothing ID', 'Review Text', 'Recommended IND']]
data.columns = ['EmployeeID', 'Review Text', 'Recommend']


# In[7]:


# Inspecting the variables
data.info()


# In[8]:


# Replacing blank variables with 'unknown' ready for processing
data['Review Text'].fillna('unknown', inplace=True)


# In[9]:


# Importing SKLearn's list of stopwords and then appending with my own words 
stop = text.ENGLISH_STOP_WORDS

# Basic text cleaning function
def remove_noise(text):
    
    # Make lowercase
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    # Remove whitespaces
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    
    # Remove special characters
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    
    # Remove punctuation
    text = text.str.replace('[^\w\s]', '')
    
    # Remove numbers
    text = text.str.replace('\d+', '')
    
    # Remove Stopwords
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    # Convert to string
    text = text.astype(str)
        
    return text


# In[10]:


# Applying noise removal function to data
data['Filtered Review Text'] = remove_noise(data['Review Text'])
data.head()


# In[11]:


# Defining a sentiment analyser function
def sentiment_analyser(text):
    return text.apply(lambda Text: pd.Series(TextBlob(Text).sentiment.polarity))

# Applying function to reviews
data['Polarity'] = sentiment_analyser(data['Filtered Review Text'])
data.head(10)


# In[14]:


import nltk
nltk.download('wordnet')

# Instantiate the Word tokenizer & Word lemmatizer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Define a word lemmatizer function
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

# Apply the word lemmatizer function to data
data['Filtered Review Text'] = data['Filtered Review Text'].apply(lemmatize_text)
data.head()


# In[15]:


# Getting a count of words from the documents
# Ngram_range is set to 1,2 - meaning either single or two word combination will be extracted
cvec = CountVectorizer(min_df=.005, max_df=.9, ngram_range=(1,2), tokenizer=lambda doc: doc, lowercase=False)
cvec.fit(data['Filtered Review Text'])


# In[16]:


# Getting the total n-gram count
len(cvec.vocabulary_)


# In[17]:


# Creating the bag-of-words representation
cvec_counts = cvec.transform(data['Filtered Review Text'])
print('sparse matrix shape:', cvec_counts.shape)
print('nonzero count:', cvec_counts.nnz)
print('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))


# In[18]:


# Instantiating the TfidfTransformer
transformer = TfidfTransformer()

# Fitting and transforming n-grams
transformed_weights = transformer.fit_transform(cvec_counts)
transformed_weights


# In[19]:


# Getting a list of all n-grams
transformed_weights = transformed_weights.toarray()
vocab = cvec.get_feature_names()

# Putting weighted n-grams into a DataFrame and computing some summary statistics
model = pd.DataFrame(transformed_weights, columns=vocab)
model['Keyword'] = model.idxmax(axis=1)
model['Max'] = model.max(axis=1)
model['Sum'] = model.drop('Max', axis=1).sum(axis=1)
model.head(10)


# In[20]:


# Merging td-idf weight matrix with original DataFrame
model = pd.merge(data, model, left_index=True, right_index=True)


# In[21]:


# Printing the first 10 reviews left
model.head(10)


# In[22]:


# Getting a view of the top 20 occurring words
occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'Term': cvec.get_feature_names(), 'Occurrences': occ})
counts_df.sort_values(by='Occurrences', ascending=False).head(25)


# In[23]:


# Getting a view of the top 20 weights
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'Term': cvec.get_feature_names(), 'Weight': weights})
weights_df.sort_values(by='Weight', ascending=False).head(25)


# In[24]:


# Plotting overall recommendations and getting value counts
fig = plt.figure(figsize = (10,5))
sns.countplot(x='Recommend', data = model)

print(data['Recommend'].value_counts())


# In[25]:


# Visualising polarity between recommending and non-recommending customers, then getting value counts
g = sns.FacetGrid(model, col="Recommend", col_order=[1, 0])
g = g.map(plt.hist, "Polarity", bins=20, color="g")

recommend = model.groupby(['Recommend'])
recommend['Polarity'].mean()


# In[26]:


# Get a list of columns for deletion
model.columns


# In[27]:


# Drop all columns not part of the text matrix
ml_model = model.drop(['EmployeeID', 'Review Text', 'Filtered Review Text', 'Polarity', 'Keyword', 'Max', 'Sum'], axis=1)

# Create X & y variables for Machine Learning
X = ml_model.drop('Recommend', axis=1)
y = ml_model['Recommend']

# Create a train-test split of these variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


# In[28]:


# Defining a function to fit and predict ML algorithms
def model(mod, model_name, x_train, y_train, x_test, y_test):
    mod.fit(x_train, y_train)
    print(model_name)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 5)
    predictions = cross_val_predict(mod, X_train, y_train, cv = 5)
    print("Accuracy:", round(acc.mean(),3))
    cm = confusion_matrix(predictions, y_train)
    print("Confusion Matrix:  \n", cm)
    print("                    Classification Report \n",classification_report(predictions, y_train))


# In[29]:


# 1. Gaussian Naive Bayes
gnb = GaussianNB()
model(gnb, "Gaussian Naive Bayes", X_train, y_train, X_test, y_test)


# In[30]:


# 2. Random Forest Classifier
ran = RandomForestClassifier(n_estimators=50)
model(ran, "Random Forest Classifier", X_train, y_train, X_test, y_test)


# In[31]:


# 3. Logistic Regression
log = LogisticRegression()
model(log, "Logistic Regression", X_train, y_train, X_test, y_test)


# In[32]:


# 4. Linear SVC
svc = LinearSVC()
model(svc, "Linear SVC", X_train, y_train, X_test, y_test)


# In[33]:



#algorithm 2
# Import the hopeful solution to our problems
from imblearn.over_sampling import SMOTE
smote=SMOTE()


# In[34]:


# Setting up new variables for ML
X_sm, y_sm = smote.fit_sample(X,y)

X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, test_size=0.3, random_state=100)


# In[35]:


# Defining a new function with revised inputs for the new SMOTE variables
def model_sm(mod, model_name, x_train_sm, y_train_sm, x_test_sm, y_test_sm):
    mod.fit(x_train_sm, y_train_sm)
    print(model_name)
    acc = cross_val_score(mod, X_train_sm, y_train_sm, scoring = "accuracy", cv = 5)
    predictions = cross_val_predict(mod, X_train_sm, y_train_sm, cv = 5)
    print("Accuracy:", round(acc.mean(),3))
    cm = confusion_matrix(predictions, y_train_sm)
    print("Confusion Matrix:  \n", cm)
    print("                    Classification Report \n",classification_report(predictions, y_train_sm))


# In[36]:


# 1. Gaussian Naive Bayes
gnb = GaussianNB()
model_sm(gnb, "Gaussian Naive Bayes", X_train_sm, y_train_sm, X_test_sm, y_test_sm)


# In[37]:


# 2. Random Forest Classifier
ran = RandomForestClassifier(n_estimators=50)
model_sm(ran, "Random Forest Classifier", X_train_sm, y_train_sm, X_test_sm, y_test_sm)


# In[38]:


# 3. Logistic Regression
log = LogisticRegression()
model_sm(log, "Logistic Regression", X_train_sm, y_train_sm, X_test_sm, y_test_sm)


# In[39]:


# 4. Linear SVC
svc = LinearSVC()
model_sm(svc, "Linear SVC", X_train_sm, y_train_sm, X_test_sm, y_test_sm)


# In[40]:


# Creating a plot for feature importance
def importance_plotting(data,x,y,palette,title):
    sns.set(style="whitegrid")
    ft = sns.PairGrid(data,y_vars=y,x_vars=x,size=5,aspect=1)
    ft.map(sns.stripplot,orient='h',palette=palette, edgecolor="black",size=15)
    for ax, title in zip(ft.axes.flat, titles):
        
    # Set a different title for each axes
        ax.set(title=title)
        
    # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    plt.show()


# In[41]:


# Compile arrays of columns (words) and feature importances
fi = {'Words':ml_model.drop('Recommend',axis=1).columns.tolist(),'Importance':ran.feature_importances_}

# Bung these into a dataframe, rank highest to lowest then slice top 20
Importance = pd.DataFrame(fi,index=None).sort_values('Importance',ascending=False).head(25)

# Plot the graph!
titles = ["Top 25 most important words in predicting product recommendation"]
importance_plotting(Importance,'Importance','Words','Greens_r',titles)


# In[ ]:




