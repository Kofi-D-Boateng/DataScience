# Linear Regression Project

# Project Abstract

This project focus is on market data for an Ecommerce company and whether they should focus on their website or mobile app develop to retain market share.

# Dataset

The dataset is a csv contains data from an Econmmerce customer base. The data tracked in the csv is as follow, but not limited to:

- Average Session Length: Average session of in-store style advice sessions
- Time on app: Average time spent on App in minutes
- Time on Website: Average time spent on Website in minutes
- Length of Membership: How many years the customer has been a member
- Email: email of the customer

# Data Insight

After cleaning the data and generating visuals, we begin to see a few trends.

- [The correlation between time spent on the website and yearly amount spent is low.](./WebsiteTimeVsAmountSpent.png)
- [There is a linear correlation that can be seen between time spent on the app and yearly amount spent.](./AppTimeVsAmountSpent.png)
- [There is linear correlation between the length of a membership and the yearly amount spent.](./AmountSpentVsMembership.png)

Now, we will look at the results of our model after training

# Inferring from the data

The results listed below are from the linear regression model

- [Model Testing](./LinearRegression.png)
- [Prediction Coefficients](./predicted-number.csv)

Based on the coefficients we found, for every n+1 increase in Time on App, the company receives $38.59 versus just $0.19 dollars via the website.

# Conclusion

The conclusion I can draw from the results, the company should invest more into the app, which also had a viable linear correlation with an increase in revenue for the company.
