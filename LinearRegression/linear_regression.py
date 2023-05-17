
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# 
# ** Read in the Ecommerce Customers csv file as a DataFrame called customers.**

customers = pd.read_csv("Ecommerce Customers")

# customers.head()

# customers.info()

# customers.describe()

sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=customers)
# plt.savefig("WebsiteTimeVsAmountSpent.png", dpi=300, transparent=False)

sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=customers)
# plt.savefig("AppTimeVsAmountSpent.png", dpi=300, transparent=False)

sns.jointplot(x="Time on Website", y="Yearly Amount Spent", kind="hex",data=customers)

sns.pairplot(data=customers)
# plt.savefig("pairplot.png", dpi=300)

sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=customers)
# plt.savefig("AmountSpentVsMembership.png", dpi=300, transparent=False)

# Training and Testing Data

X = customers[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]
y = customers["Yearly Amount Spent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



# Training the Model and fitting the model

lm: LinearRegression = LinearRegression() 

lm.fit(X_train, y_train)
print(lm.coef_)


# ## Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# ** Use lm.predict() to predict off the X_test set of the data.**

lm.predict(X_test)

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.savefig("LinearRegression.png")


# Evaluating the Model

print("MAE:", metrics.mean_absolute_error(y_test, predictions))
print("MSE:", metrics.mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

sns.distplot((y_test-predictions), bins=50)

df = pd.DataFrame(lm.coef_, X.columns)
df.columns = ["Coefficients"]
# df.to_csv("predicted-number.csv")
