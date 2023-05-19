
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

loans = pd.read_csv("loan_data.csv")

# loans.info()

# loans.describe()

# loans.head()


#  Exploratory Data Analysis

def histograph_generation(context: str):
    plt.figure(figsize=(10,6))
    loans[loans[context]== 1]["fico"].hist(alpha=0.5,color='blue', bins=30,label=f'{context}=1')
    loans[loans[context]== 0]["fico"].hist(alpha=0.5,color='red', bins=30,label=f'{context}=0')
    plt.legend()
    plt.xlabel("FICO Score")
    plt.ylabel("Count")
    name = '_'.join(context.split("."))
    plt.savefig(name,dpi=300,facecolor="white", bbox_inches="tight", transparent=False)

contexts = ["credit.policy","not.fully.paid"]

for context in contexts: histograph_generation(context)

plt.figure(figsize=(11,7))
sns.countplot(x="purpose", data=loans, hue="not.fully.paid", palette="Set1")

plt.tight_layout()
plt.savefig("LoansByPurpose.png",dpi=300,facecolor="white", bbox_inches="tight", transparent=False)


sns.jointplot(x="fico", y="int.rate", data=loans, color="purple")
plt.savefig("FICOJointPlot.png",dpi=300,facecolor="white", bbox_inches="tight", transparent=False)

sns.lmplot(x="fico", y="int.rate", data=loans, col="not.fully.paid", hue="credit.policy", palette="Set1")

plt.legend()
plt.savefig("DecisionTreeComparison.png",dpi=300,facecolor="white", bbox_inches="tight", transparent=False)

# Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.


cat_feat = ["purpose"]

final_data = pd.get_dummies(loans,columns=cat_feat,drop_first=True)

final_data.head()

# Train Test Split

X = final_data.drop("not.fully.paid", axis=1)

y= final_data["not.fully.paid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# Training a Decision Tree Model

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

# ## Predictions and Evaluation of Decision Tree

pred = dtree.predict(X_test)

print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

# Training the Random Forest model

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

# Predictions and Evaluation

rf_pred = rf.predict(X_test)

print(classification_report(y_test,rf_pred))

print(confusion_matrix(y_test,rf_pred))

def create_report(test_pred_tuple:tuple):
    with open(test_pred_tuple[2],'w') as f:
        f.write(classification_report(test_pred_tuple[0],test_pred_tuple[1]))

tuple_arr = [(y_test,pred,"decisiontree_classification.txt"),(y_test,rf_pred,"randomforest_classification.txt")]

for tuple in tuple_arr: create_report(tuple)