# Logistic Regression Project

# Project Abstract

This project focus is on advertising data and classifying when a person will click on a ad based on features in the ad and the person.

# Dataset

The dataset a csv of advertising and personal data features. The features are the following:

- Daily Time Spent on Site : consumer time on site in minutes
- Age : cutomer age in years
- Area Income : Avg. Income of geographical area of consumer
- Daily Internet Usage : Avg. minutes a day consumer is on the internet
- Ad Topic Line : Headline of the advertisement
- City : City of consumer
- Male : Whether or not consumer was male
- Country : Country of consumer
- Timestamp : Time at which consumer clicked on Ad or closed window
- Clicked on Ad : 0 or 1 indicated clicking on Ad

# Data Insight

After visualizing the data, here are some of the following trends we can infer

- [The age demographics suggest that we have more insight for approx. the 25 to about 45 age range](./Age_Histogram.png)
- [People who spent more time on the sight were individuals who usually spend a lot of time on the interet](./DailyUsage_DailyTime.png)
- [The age demographic for individuals in higher income areas falls roughly between 25 and 45](./Age_AreaIncome.png)
- [Extra visuals using a pairplot based on Clicked on Ad](./Ad_pairplot.png)

# Inferring on the data

Here are the results from the training of the model
[Classification Report](./ad_classification_report.txt)

Looking at the results, the model was able to score pretty efficiently on predicting when a user would click on an add based on the parameters chosen with a f1-score of 91% and an increase in precision after the first trial run form 86% to 96%.

# Conclusion

We can conclude that the model fit the criteria we were looking to accomplish, however it could possibly be overfit as the precision is too high after a second run on the data. If the algorithm holds up, then we will have a good model to test in the wild.
