# K-Nearest Neighbors

# Project Abstract

This project uses a masked data set, similar to one given during an interview for a data science job, and we are tasked to group these groups of data points into their respective groups via a label called "TARGET CLASS".

# Dataset

The dataset used is a masked dataset with one data point of value:

- Target Class: A binary indicator

# Data Insight

Due to the fact that the data is randomized, we decided to use a pairplot that will look at the data as a whole while using the target class column as the "hue"

- [Pair plot](./pairplot.png)

# Tuning the model

For tuning the model, we used a concept called the elbow method. The elbow method allows for scientist to chose amongst the most optimal K value by running the model through a range of K values and plotting them against the error rate. As the K value increases, eventually the model will settle into a range were any increase in K will result in neglible gains or even result in overfitting.

# Inferring on the data and conclusion

The purpose of this project was to garner an understanding of how KNearstNeighbors can group objects into "buckets" or "likeness" based off of similar features, basically classification. Therefore, the conclusion gathered is that our algorithm performed decent with the basic features given due using the elbow method concept

- [Results](./knn_classification.txt)
