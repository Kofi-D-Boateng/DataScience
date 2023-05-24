
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report


df = pd.read_csv("KNN_Project_Data")

sns.pairplot(data=df,hue="TARGET CLASS")
# plt.savefig("pairplot.png",dpi=300,facecolor="white",bbox_inches="tight",transparent=False)

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scalar_feature = scaler.transform(df.drop('TARGET CLASS',axis=1))

df_feat = pd.DataFrame(scalar_feature,columns=df.columns[:-1])
df_feat.head()

x_train, x_test,y_train,y_test = train_test_split(scalar_feature,df['TARGET CLASS'],test_size=0.30)

# Elbow method to find best K value

errors = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred = knn.predict(x_test)
    errors.append(np.mean(pred != y_test))
print(errors)

plt.figure(figsize=(10,6))
plt.plot(range(1,40),errors,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# Choose best K value against the error rate after using the elbow method.

knn = KNeighborsClassifier(n_neighbors=26)

knn.fit(x_train,y_train)

pred = knn.predict(x_test)

print(confusion_matrix(y_test,pred))
print("\n")
print(classification_report(y_test,pred))

def create_report(test_pred_tuple:tuple):
    with open(test_pred_tuple[2],'w') as f:
        f.write(classification_report(test_pred_tuple[0],test_pred_tuple[1]))

tuple_arr = [(y_test,pred,"knn_classification.txt")]

for tuple in tuple_arr: create_report(tuple)