
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_csv("College_Data", index_col=0)

# df.head()

# df.info()

# df.describe()

sns.scatterplot(x="Room.Board", y="Grad.Rate", data=df, hue="Private")
# plt.savefig("Room&BoardCostVsGradRate.png", dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)

sns.scatterplot(x="Outstate", y="F.Undergrad", data=df, hue="Private")
# plt.savefig("FullTimeStudentVsOutOfState.png", dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
# plt.savefig("SchoolType", dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
# plt.savefig("GradRateOfSchoolType", dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)

names = df[df["Grad.Rate"] > 100].columns

for name in names:
    df['Grad.Rate'][name] = 100



sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=30,alpha=0.8)
# plt.savefig("NewGradRate.png", dpi=300, facecolor="white", bbox_inches="tight" ,transparent=False)

kmean = KMeans(n_clusters=2)

kmean.fit(df.drop("Private", axis=1))

print(kmean.cluster_centers_)


# Evaluation

# There is no perfect way to evaluate clustering if you don't have the labels, however, we do have the labels, so we take advantage of this to evaluate our clusters, keep in mind, you usually won't have this luxury in the real world.
#  Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.
def Convert(cluster):
    if cluster=="Yes":
        return 1
    else:
        return 0

df["Cluster"] = df["Private"].apply(Convert)


df.head()


# Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.

print(confusion_matrix(df['Cluster'],kmean.labels_))
print(classification_report(df['Cluster'],kmean.labels_))
