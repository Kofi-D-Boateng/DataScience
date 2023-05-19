# Decision Tree & Random Forest Project

# Project Abstract

This is project focuses is on a bank or loaning entity that wants to use a machine learning model to help with considering eligibility for loans. This model will be tasked with determining whether a party was able to pay off their loan based of features in the data.

We will explore publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public. We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full.

# Dataset

Here are some of the features present in the data:

- Credit Policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
- Purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
- Interest Rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
- Installment: The monthly installments owed by the borrower if the loan is funded.
- Natural Log Annual Income: The natural log of the self-reported annual income of the borrower.
- Debt to Income: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
- Fico: The FICO credit score of the borrower.
- Days With Credit Line: The number of days the borrower has had a credit line.
- Revolving Balance: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
- Revolving Utilization: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
- Inquries Last 6 months: The borrower's number of inquiries by creditors in the last 6 months.
- Delinquencies 2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
- Public Record: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# Data Insight

After cleaning the data and creating some visualizations, certain trends began to emerge.

- [Higher FICO scores tended to have to meet the criteria for the lending club](./credit_policy.png)
- [The most sought after purpose for a loan was debt consolidation and is the highest non-paid off loan](./LoansByPurpose.png)
- [As FICO score increases, the interest rate on the loan is usually lower](./FICOJointPlot.png)
- [An high majority of people tended to pay of their loans across all FICO scores](./not_fully_paid.png)
- [There is a decreasing linear correlation between higher FICO scores, meeting the club's policy (denoted as 1) and having a lower interest rate](./DecisionTreeComparison.png)

# Inferring on the Results

After creating the model, I ran the data through a Decision Tree and Random Forest algorithm. Here are the following results.

- [Decision Tree](./decisiontree_classification.txt)
- [Random Forest](./randomforest_classification.txt)

# Conclusion

Based on the reports generated, the model did not hold up well against the data during the second phase as the recall rate was extremely low. Therefore, more engineering of the data could be done to make the model better.
