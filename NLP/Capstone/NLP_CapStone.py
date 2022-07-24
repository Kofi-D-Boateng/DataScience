import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ## The Data
# 
# **Read the yelp.csv file and set it as a dataframe called yelp.**
yelp = pd.read_csv("yelp.csv")


# ** Check the head, info , and describe methods on yelp.**
yelp.head()

yelp.info()

yelp.describe()


# **Create a new column called "text length" which is the number of words in the text column.**

yelp['text length'] = yelp['text'].apply(len)
yelp['text length']

# # EDA
# 
# Let's explore the data
# **Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings. Reference the seaborn documentation for hints on this**

g = sns.FacetGrid(yelp, col="stars")
g.map(plt.hist, 'text length')


# **Create a boxplot of text length for each star category.**
sns.boxplot(x='stars', y='text length', data=yelp, palette='rainbow')


# **Create a countplot of the number of occurrences for each type of star rating.**

sns.countplot(data=yelp, x="stars", palette='rainbow')


# ** Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation:**
yelp1=yelp.groupby('stars').mean()
yelp1


# **Use the corr() method on that groupby dataframe to produce this dataframe:**
yelp1.corr()


# **Then use seaborn to create a heatmap based off that .corr() dataframe:**
sns.heatmap(yelp1.corr(),cmap='coolwarm', annot=True)


# ## NLP Classification Task
# 
# Let's move on to the actual task. To make things a little easier, go ahead and only grab reviews that were either 1 star or 5 stars.
# 
# **Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.**
yelp_class = yelp[(yelp.stars== 1) | (yelp.stars== 5)]


# ** Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)**
X = yelp_class["text"]

y = yelp_class['stars']


# **Import CountVectorizer and create a CountVectorizer object.**
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# ** Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.**
X = cv.fit_transform(X)


# ## Train Test Split
# 
# ** Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101 **

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training a Model
# ** Import MultinomialNB and create an instance of the estimator and call is nb **

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# **Now fit nb using the training data.**
nb.fit(X_train,y_train)


# ## Predictions and Evaluations
# 
# Time to see how our model did!
# 
# **Use the predict method off of nb to predict labels from X_test.**
predictions = nb.predict(X_test)


# ** Create a confusion matrix and classification report using these predictions and y_test **

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,predictions))
print("\n")
print(classification_report(y_test,predictions))


# Let's see what happens if we try to include TF-IDF to this process using a pipeline.

# # Using Text Processing
# 
# ** Import TfidfTransformer from sklearn. **

from sklearn.feature_extraction.text import TfidfTransformer


# ** Import Pipeline from sklearn. **
from sklearn.pipeline import Pipeline


# ** Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()**
pipeline = Pipeline([
    ("bow: ", CountVectorizer()),
    ("tfidf: ", TfidfTransformer()),
    ("classifier: ", MultinomialNB())
])


# ## Using the Pipeline
# 
# **Time to use the pipeline! Remember this pipeline has all your pre-process steps in it already, meaning we'll need to re-split the original data (Remember that we overwrote X as the CountVectorized version. What we need is just the text**

# ### Train Test Split
# 
# **Redo the train test split on the yelp_class object.**
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# **Now fit the pipeline to the training data. Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels**


pipeline.fit(X_train,y_train)


# ### Predictions and Evaluation
# 
# ** Now use the pipeline to predict from the X_test and create a classification report and confusion matrix. You should notice strange results.**
predict = pipeline.predict(X_test)

print(confusion_matrix(y_test,predict))
print("\n")
print(classification_report(y_test,predict))


# Looks like Tf-Idf actually made things worse! That is it for this project.