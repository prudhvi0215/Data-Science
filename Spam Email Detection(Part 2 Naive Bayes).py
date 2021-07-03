#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[39]:


spam_df = pd.read_csv('spam.csv')
spam_df.head()


# In[40]:


spam_df.groupby('Category').describe()


# In[41]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[42]:


spam_df.Category = le.fit_transform(spam_df.Category)
spam_df.head()


# In[43]:


from sklearn.pipeline import Pipeline


# In[44]:


# model = Pipeline([
#     ('vectorizer',CountVectorizer()),
#     ('nb',MultinomialNB())
# ])


# In[45]:


X_train,X_test,y_train,y_test = train_test_split(spam_df['Message'],spam_df['Category'],test_size=0.2)


# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
x_train_count = vec.fit_transform(X_train.values)
x_train_count.toarray()[:2]


# In[47]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_count,y_train)


# In[50]:


x_test_count = vec.transform(X_test)
model.score(x_test_count,y_test)


# In[49]:


emails=[
    "Hurray! You Won $1000",
    "Hi, My name is prudhvi, can we meet tommorrow"
]
    
emails_count = vec.transform(emails)
model.predict(emails_count)

