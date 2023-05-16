# # 911 Calls Capstone Project

# For this capstone project we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 

# ## Data and Setup

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("911.csv")

df.info()

df.head()


# ## Basic Questions

# ** What are the top 5 zipcodes for 911 calls? **
# df["zip"].value_counts().head(5)

# ** What are the top 5 townships (twp) for 911 calls? **
# df["twp"].value_counts().head(5)

# ** Take a look at the 'title' column, how many unique title codes are there? **
# df["title"].nunique()

# ## Creating new features
df["Reason"] = df["title"].apply(lambda title: title.split(":")[0])


# ** What is the most common Reason for a 911 call based off of this new column? **
# df["Reason"].value_counts().head(3)

# ** Now use seaborn to create a countplot of 911 calls by Reason. **
sns.countplot(x="Reason", data=df, palette='viridis')

# plt.savefig("Reasons.png", dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)
# type(df["timeStamp"].iloc[0])

df["timeStamp"] = pd.to_datetime(df["timeStamp"])

df["Hour"]= df["timeStamp"].apply(lambda time: time.hour )
df["Month"]= df["timeStamp"].apply(lambda time: time.month )
df["Day of Week"]= df["timeStamp"].apply(lambda time: time.dayofweek)


df["Day of Week"] = df["Day of Week"].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})


sns.countplot(x="Day of Week", data =df, hue="Reason", palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# plt.savefig("ReasonsPerWeek.png", dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)

sns.countplot(x="Month", data =df, hue="Reason", palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# plt.savefig("ReasonsPerMonth.png", dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)

byMonth = df.groupby('Month').count()
# byMonth.head()
# byMonth['twp'].plot()


sns.lmplot(x="Month", y="twp", data=byMonth.reset_index())
# plt.savefig("TwpVsMonthLF.png", dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)



df['Date']= df["timeStamp"].apply(lambda t: t.date())

t= df.groupby('Date').count()['twp']
plt.figure(figsize=(10,5))
plt.title("Timestamp of emergencies")
plt.plot(t)
plt.tight_layout()
# plt.savefig("TimestampsOfEmergencies.png", dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)

def callsByReason(reason: str) -> None:
    t = df[df["Reason"]== f'{reason}'].groupby('Date').count()['twp']
    plt.figure(figsize=(10,5))
    plt.title(f'{reason}')
    plt.plot(t)
    plt.tight_layout()
    # plt.savefig(f'{reason}Calls.png', dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)

reasons = ["Traffic","Fire","EMS"]
for r in reasons: callsByReason(r)


#  Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method.

def week_hour_heatmaps(interval: str) -> None:
    i = df.groupby(by=["Day of Week", f'{interval}']).count()['Reason'].unstack()

    plt.figure(figsize=(12,6))
    sns.heatmap(i, cmap='viridis')
    plt.savefig(f'{interval}Heatmap.png', dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)
    sns.clustermap(i, cmap='viridis')

intervals = ["Hour","Month"]

for i in intervals: week_hour_heatmaps(i)
