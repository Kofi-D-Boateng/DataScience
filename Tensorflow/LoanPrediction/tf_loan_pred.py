# Tensorflow Project

# ## The Data
# 
# We will be using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club
# 
# LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California.[3] It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.

# ### Our Goal
# 
# Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), can we build a model thatcan predict wether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan. Keep in mind classification metrics when evaluating the performance of your model!
# 
# The "loan_status" column contains our label.


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import random
from sklearn.metrics import classification_report


# data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')


# print(data_info.loc['revol_util']['Description'])


# def feat_info(col_name):
#     print(data_info.loc[col_name]['Description'])


# feat_info('mort_acc')

print('[IN PROGRESS]: Loading data......')
df = pd.read_csv('./lending_club_loan_two.csv')

# df.info()

# Exploratory Data Analysis
# OVERALL GOAL: Get an understanding for which variables are important, view summary statistics, and visualize the data

# sns.countplot(x="loan_status", data=df)

# plt.figure(figsize=(10,6))
# sns.histplot(df["loan_amnt"], kde=False, bins=70)

# plt.figure(figsize=(12,7))
# sns.heatmap(df.select_dtypes(include='number').corr(),cmap='viridis', annot=True)
# plt.ylim(10, 0)
# plt.title("Correlation Heat Map")

#  You should have noticed almost perfect correlation with the "installment" feature. Explore this feature further. Print out their descriptions and perform a scatterplot between them. Does this relationship make sense to you? Do you think there is duplicate information here?

# feat_info('installment')

# feat_info('loan_amnt')

# sns.scatterplot(x="installment",y="loan_amnt", data=df)

# Calculate the summary statistics for the loan amount, grouped by the loan_status.**

# df.groupby("loan_status")["loan_amnt"].describe().transpose()

# Let's explore the Grade and SubGrade columns that LendingClub attributes to the loans. What are the unique possible grades and subgrades?

# def sorting(col: str, dataframe: pd.DataFrame):
#     return sorted(dataframe[col].unique())

# cols = ['grade','sub_grade']

# for col in cols: 
#     r = sorting(col,df)
#     print(r)

# Create a countplot per grade. Set the hue to the loan_status label.

# sns.countplot(x="grade",hue="loan_status",data=df)

# Display a count plot per subgrade. You may need to resize for this plot and [reorder](https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot) the x axis. Feel free to edit the color palette. Explore both all loans made per subgrade as well being separated based on the loan_status. After creating this plot, go ahead and create a similar plot, but set hue="loan_status"**

# plt.figure(figsize=(12,4))
# subgrade_order = sorted(df['sub_grade'].unique())
# sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm',hue='loan_status')

#  It looks like F and G subgrades don't get paid back that often. Isloate those and recreate the countplot just for those subgrades.**


# f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

# plt.figure(figsize=(12,4))
# subgrade_order = sorted(f_and_g['sub_grade'].unique())
# sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue="loan_status")

df["loan_repaid"] = df["loan_status"].map({"Fully Paid": 1,"Charged Off":0})

# df[['loan_repaid','loan_status']]


# Create a bar plot showing the correlation of the numeric features to the new loan_repaid column.
# df.corr(numeric_only=True)["loan_repaid"].sort_values().drop("loan_repaid").plot(kind="bar")


# Data PreProcessing

print('[IN PROGRESS]: Beginning data preprocessing......')

# # Missing Data
# Let's explore this missing data columns.

# df['emp_title'].nunique()

# df['emp_title'].value_counts()

# Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column.**

df.drop("emp_title",axis=1,inplace=True)

# Create a count plot of the emp_length feature column. Challenge: Sort the order of the values.

# emp_length_order = [ '< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years']

# plt.figure(figsize=(12,6))
# sns.countplot(x="emp_length", data = df, order=emp_length_order)

# Plot out the countplot with a hue separating Fully Paid vs Charged Off

# plt.figure(figsize=(12,4))
# sns.countplot(x="emp_length",order=emp_length_order,hue="loan_status", data=df)

# emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']

# emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']

# emp_len = emp_co/emp_fp

# emp_len

# emp_len.plot(kind="bar")


# Charge off rates are extremely similar across all employment lengths. Drop the emp_length column.

df.drop('emp_length', axis=1,inplace=True)

# The title column is simply a string subcategory/description of the purpose column. Go ahead and drop the title column.

df.drop("title", axis=1,inplace=True)

# **CHALLENGE TASK: Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above. This involves using an .apply() method with two columns. Check out the link below for more info, or review the solutions video/notebook.**

def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc
    else:
        return mort_acc


df["mort_acc"] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)


# revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data. Go ahead and remove the rows that are missing those values in those columns with dropna().

df.dropna(inplace=True)


# Categorical Variables and Dummy Variables

df['term'] = df['term'].apply(lambda term: int(term[:3]))

# grade feature
# 
# We already know grade is part of sub_grade, so just drop the grade feature.**

df.drop("grade", axis=1,inplace=True)

# One-Hot Encoding

print('[IN PROGRESS]: Performing data encoding......')

# Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe. Remember to drop the original subgrade column and to add drop_first=True to your get_dummies call.

def one_hot_encoder(encoding_cols: list, dataframe: pd.DataFrame,drop_first_flag: bool) -> pd.DataFrame:
    dummy_list: List[pd.DataFrame] = []
    for col in encoding_cols:
        dummy_list.append(pd.get_dummies(dataframe[col],drop_first=drop_first_flag))
    df: pd.DataFrame = pd.DataFrame()
    for i in range(0,len(dummy_list)):
        df = pd.concat([dataframe.drop(encoding_cols[i], axis=1),dummy_list[i]], axis=1)
    return df

# Convert these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.**

# dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
# df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
# df = pd.concat([df,dummies],axis=1)

cols = ["sub_grade",'verification_status', 'application_type','initial_list_status','purpose']

df = one_hot_encoder(cols,df,True)

#  Convert these to dummy variables

# Home ownership engineering

df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

# zip code engineering

df["zip_code"] = df["address"].apply(lambda address: address[-5:])

dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)


# issue_d 

#  This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature.**

df.drop("issue_d", axis=1, inplace=True)

# ### earliest_cr_line
#  This appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.**

df["earliest_cr_year"] = df["earliest_cr_line"].apply(lambda year: int(year[-4:]))

df.drop("earliest_cr_line", axis=1,inplace=True)


#  Train Test Split

print('[IN PROGRESS]: Beginning training......')

#  drop the load_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s.**

df.drop("loan_status", axis=1,inplace=True)

# X = df.drop('loan_repaid', axis=1).values
X = df.select_dtypes('number').drop('loan_repaid',axis=1).values

y = df["loan_repaid"].values

# OPTIONAL
# Grabbing a Sample for Training Time
# OPTIONAL: Use .sample() to grab a sample of the 490k+ entries to save time on training. Highly recommended for lower RAM computers or if you are not using GPU.

# print(len(df))

# df = df.sample(frac=0.1, random_state=101)

#  Perform a train/test split with test_size=0.2 and a random_state of 101.**

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
# print(type(y_test))

# Normalizing the Data
#  Use a MinMaxScaler to normalize the feature data X_train and X_test. Recall we don't want data leakge from the test set so we only fit on the X_train data.**

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# # Creating the Model
# [HELP](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)
#  Build a sequential model to will be trained on the data. You have unlimited options here, but here is what the solution uses: a model that goes 78 --> 39 --> 19--> 1 output neuron. OPTIONAL: Explore adding [Dropout layers](https://keras.io/layers/core/) [1](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) [2](https://towardsdatascience.com/machine-learning-part-20-dropout-keras-layers-explained-8c9f6dc4c9ab)**

model = tf.keras.models.Sequential()

def generate_model(model: tf.keras.models.Sequential, creation_tuple: List[Tuple[int,str,float]]) -> tf.keras.models.Sequential:
    for tuple in creation_tuple:
        if tuple[2] <= 0.0:
            model.add(tf.keras.layers.Dense(units=tuple[0], activation=tuple[1]))
            model.add(tf.keras.layers.Dropout(tuple[2]))
        else:
            model.add(tf.keras.layers.Dense(units=tuple[0], activation=tuple[1]))
    return model

ct = [(len(df.columns),'relu',0.2),((random.randint(1,len(df.columns)) * 2/3) + 1,'relu',0.2),(1,'sigmoid',0.0)]

model = generate_model(model,ct)

model.compile(loss="binary_crossentropy", optimizer="adam")

# Remember to compile()

#  Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting. Optional: add in a batch_size of 256.

model.fit(x=X_train,y=y_train,epochs=25,batch_size=256,validation_data=(X_test,y_test))

#  OPTIONAL: Save your model.**

model.save("loaning_model_tutorial.h5")

# # Section 3: Evaluating Model Performance.
# 
#  Plot out the validation loss versus the training loss.**

losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()
plt.savefig('Model_Performance.png',dpi=300,facecolor='white',bbox_inches='tight',transparent=False)



# Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.
import numpy as np
predictions: np.ndarray = model.predict(X_test)

binary_predictions = np.where(predictions.flatten() >= 0.5,1,0)


# rounded_pred = np.round(predictions.flatten(),3)
# for p in rounded_pred:
#     print(f'[PREDICTION]: {p}\t type: {type(p)}\n')

# print(classification_report(y_test,predictions))
with open('classification_report.txt','w') as f:
    f.write(classification_report(y_test,binary_predictions))

# df['loan_repaid'].value_counts()


#LOOK AT F1 and RECALL (The model is not as good because the model is over represented in fully_paid)

# print(confusion_matrix(y_test,predictions))


# Testing on a customer
#  Given the customer below, would you offer this person a loan?**

print('[IN PROGRESS]: Beginning prediction test...')

random_ind = random.randint(0,101)

new_customer = df.drop('loan_repaid',axis=1).select_dtypes('number').iloc[random_ind]


new_customer = scaler.transform(new_customer.values.reshape(1,39))

result = model.predict(new_customer)

result_object = {
    'result':str(result[0][0]),
    'result w/ threshold bound':str(0 if result[0] <= 0.5 else 1),
    'actual result':str(df.iloc[random_ind]["loan_repaid"])
}
print('[IN PROGRESS]: writing prediction test to file...')

with open('Results.txt','w') as f:
    f.write(json.dumps(result_object))


print(f'[COMPLETE]: Results: {result_object}')
