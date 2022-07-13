# Credit-card-fraud-detection-
Data:
We will be using the Credit Card Detection dataset from Kaggle. The data is highly
unbalanced with about 99.8% non-fraud transactions. There are 492 fraudulent transactions and
284,807 transactions in the dataset we chose. The dataset was previously transformed through
PCA for confidentiality of the consumers transactions. There are features
Features:
• V1 to V28 results from the original features transformed by PCA.
• ‘Time’: The time column shows the second between each transaction with the
first transaction. This feature was not relevant therefore we removed it from the
original creditcard dataset.
• ‘Amount’: The ‘Amount feature column is the amount of the transaction. This
can be useful to compare the fraud and non-fraud transactions and the amount
each amounted to.
• ‘Class’: The Class column is the y variable (responsive) in the dataset. It
contains 2 values, 1 is for a fraud transaction and 0 is for a non-fraud
transaction.

Observations from EDA
1. We checked for missing values because missing values in the data can be problematic and may
not yield accurate results. Upon our observation we did not find any missing values.
2. We checked for class imbalance which refers to unequal instances of different classes. Have a
look at the visualization below. We have class (0 — No fraud, 1 — fraud) on the X-axis and the
percentage of instances plotted on Y-axis. We see that our dataset is highly unbalanced with
respect to the class of interest (Fraud).
3. Plotted a box plot on transaction amount by class. Given the number of outliers, the Amount of
each transaction is more variable with non-fraud transactions than with fraud transactions, as
shown in the boxplot below. Most transactions, both legitimate and fraudulent, were of "low"
value.
4. We plotted correlation of anonymized variables with amount. We see that most of the data aspects are
unrelated. This is because most of the features were transformed using Principal Component Analysis
(PCA) technique. After propagating the true features using PCA, the features V1 through V28 are most
likely because of the Principal Components. Based on the correlation we observe that most of the data
features are not correlated. the pie chart we can conclude that fraud transactions despite being in minority
(0.17%), they adversely affect consumers and also companies since the transaction amount is huge for the
fraud transactions.
5.Notice that the number of regular transactions decreased significantly around 90,000 seconds before
increasing again around 110,000 seconds in the graph below. It is entirely possible that this can happen at
night when people buy and trade less than during the day.
On the other hand, the majority of fraudulent transactions take place around 100,000 marks, supporting
the previous theory that criminals prefer to commit fraud at midnight, with less surveillance and less
likely victims to notice the fraud. It may be.
• Resampling
Since the data set is highly unbalanced, we have adopted resampling techniques to analyze
further. All the strategies above will be used to train a single classifier using the train data set,
with class imbalance properly modified. We will create successive models using the strategy that
delivers the best ROC Area under curve score on a holdout test set. Below are the methods we
followed and finalized on one method observing the ROC value.
Down-Sampling
Up-sampling
ROSE (Random over-sampling)

While preparing the data set as the time variable was not relevant, we removed it from the data
set and scaled all the numeric variables so that the no numeric variables should be outnumbered
by some data. We split the data into training (70%) and testing (30%) data using seed function for
reproducible results. Upon fitting the above sampling methods to decision tree model, we arrived
at following results on each sampling method. We chose the sampling approach with the
maximum area under the curve (AUC) based on the study of the three sample methods since it
will deliver the best performance. For our dataset, oversampling yielded the highest AUC. Hence,
we decided to consider Up sampled data.

ML models used :
1. Logistic regression
2. Random forest
3. Naïve Bayes
4. XGBoost

Conclusion
To establish which metric(s) is/are most relevant, we must analyze the real-world implications. We
believe that the consequences of false negatives will result in large financial losses for a financial
institution. As a result, we chose sensitivity and miss rate, as well as accuracy, as the metrics for selecting
the optimal model. We observed that fraud is entirely possible that this can happen at night when people
buy and trade less than during the day. We sought to demonstrate alternative ways for dealing with
unbalanced datasets in this project, such as the fraud credit card transaction dataset, where the number of
fraudulent cases is small relative to the number of normal transactions. We've discussed why accuracy is
not the best metric for evaluating model performance, and we have used the metric AREA UNDER ROC
CURVE to see how alternative techniques of oversampling or under sampling the response variable might
improve model training. We used precision, recall and f1 score to determine the best model because
accuracy and other methods might mislead our results. We concluded that the oversampling technique
works well on the dataset and that the model performance significantly improved over the unbalanced
data. An XGBOOST model received the highest score from all models of 0.983, while both random forest
and logistic regression models scored well as well.



