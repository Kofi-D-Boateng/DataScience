# K-Means Clustering

# Project Abstract

In this project I attempted to group two universities into public or private based off the csv data given to use.

# Dataset

The dataset used is a compilation of features from colleges. Below are some of the features visible to us:

- Private : A factor with levels No and Yes indicating private or public university
- Apps : Number of applications received
- Accept : Number of applications accepted
- Enroll : Number of new students enrolled
- Top10perc : Pct. new students from top 10% of H.S. class
- Top25perc : Pct. new students from top 25% of H.S. class
- F.Undergrad : Number of fulltime undergraduates
- P.Undergrad : Number of parttime undergraduates
- Outstate : Out-of-state tuition
- Room.Board : Room and board costs
- Books : Estimated book costs
- Personal : Estimated personal spending
- PhD : Pct. of faculty with Ph.D.’s
- Terminal : Pct. of faculty with terminal degree
- S.F.Ratio : Student/faculty ratio
- perc.alumni : Pct. alumni who donate
- Expend : Instructional expenditure per student
- Grad.Rate : Graduation rate

# What is KMeans Clustering

KMeans Clustering is the attempt to partition data into unique groupings by using the nearest mean of each processed vector

# Data Insight

Below is a few observations made regarding the data (Before training with Private label)

- [There is a small linear correlation between Room and Board cost and Graduation Rate](./Room%26BoardVsGradRate.png)
- [As Out-of-State cost increase, the overwhelmingly majority of colleges were private institution](./FullTimeStudentVsOutOfState.png)
- [More students tended to graduate from private institutions as the graduation rate increased](./NewGradRate.png)

# Inferring from reuslts and Conclusion

When attempting to use KMeans Clustering, you are suppose to manipulate the data without having specific labels to validate your hypothesis, however in this
project, because there were labels, we are looking at how well the algorithm perform.

[Results](./cluster_classification_report.txt)

We can see that there was significant drop offs between the first and second trail, and the level of accuracy is pretty low, but the alogrithm worked as expected.
