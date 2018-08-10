## Naive Bayes and Logistic Regression for classification

In this project, I use Sci-kit Learn Naive Bayes and Logistic Regression model to predict whether a person is 
a Democrat or a Republican depending on their Twitter posts. 

train_newline.txt is training data, which contains 40k tweets. 
dev_newline.txt is testing data, which contains 5k tweets. 

project1.py contains 4 models: 

Multinomial Naive Bayes Unigram, Logistic Regression Bigram,  
Multinomial Naive Bayes Trigram, and Best train model. 

It prints out accuracy, top 20 features, and confusion matrix for each model. 

classify.py file takes the "train_newline.txt" and "dev_newline.txt" as command line arguments.
Type ‘python classify.py train_newline.txt test_newline.txt’ in your terminal to run. 

It saves best trained model to model.pkl, post-processed vectorized test data to test.pkl,
and CountVectorizer to features.pkl in the same directory (Will load features.pkl to get top 20 features in analyze.py file).

analyze.py file takes the best trained model "model.pkl" and verctorized test data "test.pkl" as command line arguments.

Type 'python analyze.py model.pkl test' in your terminal to run. It outputs top 20 features, confusion matrix,
and confusion matrix plot for the best trained model. 



			 



			 
