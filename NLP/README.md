# Natural Language Processing

# Abstract

The project focuses on categorization of yelp reviews into either 1 star or 5 star review based off the features given by each review.

# Dataset

The dataset given to us is a log of yelp reviews with the following features:

- Business Id: Id for a business
- Date: The created date of the review
- Review Id: The id for the specific reveiew
- Stars: Rating given by the user
- Text: Text left by the user (actual review)
- Type: The type representation for the data
- User Id: The id for the user
- Misc Features:
  - Cool, Userful, Funny

# Data Insight

During our data exploration, we were able to draw a few inferences from the data.

- [Text with 5 star ratings had the fewest words](./textLength.png)
- [There is a correlation between the count for useful tags and the text length](./Heatmap.png)

# Infering from Model Results and Conclusion

Below are the results from two model test.

- [Non TF-IDF](./non_tfidf_classification_report.txt)
- [Pipeline w/ TF-IDF](./tfidf_classification_report.txt)

Based on these results, adding TF-IDF into our pipeline do not help us predict well which review was a one star. I believe the reasoning behind this is due to the fact that there is high irregularity between the features, such as text length, word choice, and the actual rating. There is too many overlaps and therefore more feature engineering would need to be done to correct the model, if we wanted to use TF-IDF.
