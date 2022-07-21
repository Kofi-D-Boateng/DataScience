# Import necessary modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[98]:


ad_data = pd.read_csv('advertising.csv')


# **Check the head of ad_data**

# In[40]:


ad_data.head()


# ** Use info and describe() on ad_data**

# In[41]:


ad_data.info()


# In[42]:


ad_data.describe()


# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# In[48]:


sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


# **Create a jointplot showing Area Income versus Age.**

# In[64]:


sns.jointplot(x='Age',y='Area Income',data=ad_data)


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[66]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[72]:


sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')


# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[84]:


sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

# In[85]:


from sklearn.model_selection import train_test_split


# In[88]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ** Train and fit a logistic regression model on the training set.**

# In[91]:


from sklearn.linear_model import LogisticRegression


# In[92]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[94]:


predictions = logmodel.predict(X_test)


# ** Create a classification report for the model.**

# In[95]:


from sklearn.metrics import classification_report


# In[96]:


print(classification_report(y_test,predictions))


# ## Great Job!
