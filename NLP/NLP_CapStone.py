import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


yelp = pd.read_csv("yelp.csv")


# yelp.head()

# yelp.info()

# yelp.describe()


# **Create a new column called "text length" which is the number of words in the text column.**

yelp['text length'] = yelp['text'].apply(len)

# Let's explore the data

g = sns.FacetGrid(yelp, col="stars")
g.map(plt.hist, 'text length')


# **Create a boxplot of text length for each star category.**
sns.boxplot(x='stars', y='text length', data=yelp, palette='rainbow')


# **Create a countplot of the number of occurrences for each type of star rating.**

sns.countplot(data=yelp, x="stars", palette='rainbow')


# ** Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation:**
yelp_stars_avg=yelp.groupby('stars').mean()

# yelp_stars_avg.corr()


# **Then use seaborn to create a heatmap based off that .corr() dataframe:**
sns.heatmap(yelp_stars_avg.corr(),cmap='coolwarm', annot=True)


# ## NLP Classification Task
# 
# Let's move on to the actual task. To make things a little easier, go ahead and only grab reviews that were either 1 star or 5 stars.
# 
# **Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.**
yelp_class = yelp[(yelp.stars== 1) | (yelp.stars== 5)]

X = yelp_class["text"]

y = yelp_class['stars']


cv = CountVectorizer()


# ** Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.**
X = cv.fit_transform(X)


# ## Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training a Model
# ** Import MultinomialNB and create an instance of the estimator and call is nb **

nb = MultinomialNB()


# **Now fit nb using the training data.**
nb.fit(X_train,y_train)


# ## Predictions and Evaluations

predictions = nb.predict(X_test)


# ** Create a confusion matrix and classification report using these predictions and y_test **

print(confusion_matrix(y_test,predictions))
print("\n")
print(classification_report(y_test,predictions))


# Let's see what happens if we try to include TF-IDF to this process using a pipeline


# ** Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()**
pipeline = Pipeline([
    ("bow: ", CountVectorizer()),
    ("tfidf: ", TfidfTransformer()),
    ("classifier: ", MultinomialNB())
])


# ## Using the Pipeline

# ### Train Test Split
# 
# **Redo the train test split on the yelp_class object.**
X = yelp_class['text']
y = yelp_class['stars']
xx_train, xx_test, yy_train, yy_test = train_test_split(X, y, test_size=0.3, random_state=101)


# **Now fit the pipeline to the training data. Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels**


pipeline.fit(xx_train,yy_train)

# ### Predictions and Evaluation

predict = pipeline.predict(X_test)

print(confusion_matrix(y_test,predict))
print("\n")
print(classification_report(y_test,predict))