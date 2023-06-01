# Tensorflow Binary Classification Project

# Project abstract

This project is focused feature engineering of a 490K+ datasheet of lending data, to allow us to create a model to predict whether a person will repay their loan (Fully Paid) or not (Charged Off)

# Dataset

The dataset from the lending club contains the following features: [Features](./features.csv)

# Data Insight

After completing some data exploration, I was able to create some interesting data visuals from the data.

- [There is high correlation between installments and loan amounts, as well as other features within the heat map](./correlation_heatmap.png)
  - [See correlation scatterplot between installments and loan amounts](./correlation_scatterplot.png)
- [Our data is skewed towards fully paid, which will have a somewhat bias effect on the model](./loan_status.png)
- [Grade B loans are the most fully paid back loans, followed by C, then A](./loan_grades.png)
  - [Subgrade histogram](./loan_subgrades.png)

# Neural Network Model

For this project I decided to go with a [forward feeding neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) using the Keras API built into Tensorflow. Below is the setup of the model and reasoning behind specific features of the model.

- 1 input layer: The input layer will match the length of columns or features of the data set.
  - [The input layer and all subsequent layers minus the last layer, use the Rectified Linear Unit activation function](https://builtin.com/machine-learning/relu-activation-function)
- 1-2 hidden layers: For this project, I used one hidden layer, as that it is sufficent enough to predict on the data while not causing overfitting.
- 1 output layer: This layer uses the sigmoid function activation function, as we are trying to do binary classification. Therefore, we need to squash our inputs between 0 and 1.
- For compilation of the model, I used [binary crossentropy](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a) as the loss function and the [adam optimizer](https://keras.io/api/optimizers/adam/#:~:text=Adam%20optimization%20is%20a%20stochastic,order%20and%20second%2Dorder%20moments.) to pair with it.

The model will be trained on 25 epochs with a batch size of 256. What we start to notice after the training is that [As we approach 25 epochs, the prediction loss begin to stablize.](./Model_Performance.png). This would suggess to use that any epochs higher than 25 will be negligible increases in performance.

# Results and Conclusion

[The results of our model show that it does pretty well at predicting if a lendee would pay back their loans.](./classification_report.txt) However, I believe task given to this model is one that does not truly require a neural network to decide, because the data is highly skewed towards Fully-Paid off loans. Therefore, although the model predicts well on the data. A better task would be to try and predict who would not pay off their loans, and or who would be a good candidate for a loan based off the features.

[Results of a test from the model](./Results.txt)
