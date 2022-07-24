

import nltk


nltk.download_shell()


messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[ ]:


print(len(messages))


# In[33]:


messages[0]


# In[34]:


for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')


# In[35]:


import pandas as pd
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep="\t", names=["label", "messages"])


# In[36]:


messages.head()


# In[37]:


messages.describe()


# In[38]:


messages.groupby('label').describe()


# In[39]:


messages['length'] = messages['messages'].apply(len)


# In[40]:


messages.head()


# In[41]:


#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


messages['length'].plot.hist(bins=300)


# In[43]:


#finding the max length of the emails
messages['length'].describe()


# In[44]:


#pandas masking

messages[messages['length']== 910].iloc[0]


# In[45]:


#Finding difference between spam and non-spam emails
messages.hist(column="length", by='label',bins=60,figsize=(12,4))


# In[46]:


import string


# In[47]:


mess = 'Sample message! Notice: its has punctuation.'


# In[48]:


nopunc = [c for c in mess if c not in string.punctuation]


# In[49]:


from nltk.corpus import stopwords


# In[50]:


nopunc = "".join(nopunc)


# In[51]:


nopunc


# In[52]:


nopunc.split()


# In[53]:


clean_mess = [word for word in nopunc.split() if word not in stopwords.words('english')]


# In[54]:


clean_mess


# In[55]:


def text_process(mess):
    """
    1. remove punctuations
    2. remove stop words
    3. return list of clean text words
    """

    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    
    return [word for word in nopunc.split() if word not in stopwords.words('english')]


# In[56]:


messages["messages"].head(5).apply(text_process)


# In[57]:


from sklearn.feature_extraction.text import CountVectorizer


# In[58]:


bow_trans = CountVectorizer(analyzer=text_process).fit(messages["messages"])


# In[59]:


print(len(bow_trans.vocabulary_))


# In[60]:


mess4 = messages["messages"][3]
print(mess4)


# In[61]:


bow4 = bow_trans.transform([mess4])


# In[62]:


print(bow4
     )


# In[63]:


print(bow4.shape)


# In[68]:


#checking for redundant wording
bow_trans.get_feature_names()[4221]
print("\n")
bow_trans.get_feature_names()[9746]


# In[69]:


messages_bow = bow_trans.transform(messages["messages"])


# In[70]:


print("Shape of sparse matrix: ", messages_bow.shape)


# In[71]:


messages_bow.nnz


# In[73]:


#checks for number of 0 values in matrixs versus how many there actually is.

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


# In[74]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[75]:


tfidf_transform = TfidfTransformer().fit(messages_bow)


# In[76]:


tfidf = tfidf_transform.transform(bow4)


# In[77]:


print(tfidf)


# In[79]:


tfidf_transform.idf_[bow_trans.vocabulary_['university']]


# In[80]:


messages_tfidf = tfidf_transform.transform(messages_bow)


# In[81]:


from sklearn.naive_bayes import MultinomialNB


# In[82]:


spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])


# In[83]:


spam_detect_model.predict(tfidf)[0]


# In[84]:


messages['label'][3]


# In[85]:


from sklearn.model_selection import train_test_split


# In[86]:


msg_train,msg_test,label_train,label_test = train_test_split(messages["messages"],messages["label"], test_size=0.3)


# In[87]:


#Pipeline scikit learn
#allows us not to have to go through CountVectorizing again
from sklearn.pipeline import Pipeline


# In[88]:


#summarize

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


# In[89]:


pipeline.fit(msg_train,label_train)


# In[91]:


predictions = pipeline.predict(msg_test)


# In[92]:


from sklearn.metrics import classification_report


# In[93]:


print(classification_report(label_test,predictions))


# In[ ]:


#DONE

