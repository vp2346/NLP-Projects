import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys



def read_extract(filename):
    with open(filename,"r") as f:
    #with open(filename,"r",encoding="utf8") as f:
        lines=f.readlines()
    lines = [x.rstrip("\n").split("\t") for x in lines]
    X=[]
    y=[]
    for i in range(0,len(lines)):
        X.append(lines[i][0])
        y.append(lines[i][1])
    #set democrat to 0 and republican to 1
    y=[0 if i== "democrat" else 1 for i in y]  
    return X, y

X_sample=read_extract('train_newline.txt')[0]
y_train=read_extract('train_newline.txt')[1]
X_dev_sample=read_extract('dev_newline.txt')[0]
y_dev=read_extract('dev_newline.txt')[1]

#unigram
cv1 = CountVectorizer(ngram_range=(1, 1))
X_train1=cv1.fit_transform(X_sample)
X_dev1=cv1.transform(X_dev_sample)


#Unigram naive bayes'''
#Select Multinomial NB as GausssianNB only works for dense matrix. BernoulliNB is designed for binary/boolean features.
#So our data works best with MultinomialNB'''

clf_1 = MultinomialNB()
clf_1.fit(X_train1, y_train) 
y_predicted1=clf_1.predict(X_dev1)
print ("First Model: Mutinomial Naive Bayes Unigram")
print ("Accuracy is %.5f"
       % (sklearn.metrics.accuracy_score(y_dev, y_predicted1)))

features1=cv1.get_feature_names()
coefficient1=clf_1.coef_[0]

print ('Top 20 features are:')
print (sorted(zip(coefficient1, features1), reverse=True)[:20])
cm1 = confusion_matrix(y_dev, y_predicted1)
print ('Confusion Matrix:')
print(cm1)

#bigram
cv2 = CountVectorizer(ngram_range=(2, 2))
X_train2=cv2.fit_transform(X_sample)
X_dev2=cv2.transform(X_dev_sample)

#bigram logistic regression'''

clf_2 = LogisticRegression()
clf_2.fit(X_train2, y_train) 
y_predicted2=clf_2.predict(X_dev2)
print ("Second Model: Logistic Regression Bigrams")
print ("Accuracy is %.5f" 
       % (sklearn.metrics.accuracy_score(y_dev, y_predicted2)))

features2=cv2.get_feature_names()
coefficient2=clf_2.coef_[0]
print ('Top 20 features are:')
print (sorted(zip(coefficient2, features2), reverse=True)[:20])
cm2 = confusion_matrix(y_dev, y_predicted2)
print ('Confusion Matrix:')
print(cm2)

#trigram
cv3 = CountVectorizer(ngram_range=(3, 3))
X_train3=cv3.fit_transform(X_sample)
X_dev3=cv3.transform(X_dev_sample)

#trigram naive bayes multinomial NB model
clf_3 = MultinomialNB()
clf_3.fit(X_train3, y_train) 
y_predicted3=clf_3.predict(X_dev3)
print ("Third Model: Multinomial Naive Bayes Trigrams")
print ("Accuracy for Multinomial Naive Bayes Trigrams is %.5f" 
       % (sklearn.metrics.accuracy_score(y_dev, y_predicted3)))

features3=cv3.get_feature_names()
coefficient3=clf_3.coef_[0]
print ('Top 20 features are:')
print (sorted(zip(coefficient3, features3), reverse=True)[:20])
cm3 = confusion_matrix(y_dev, y_predicted3)
print ('Confusion Matrix:')
print(cm3)


#best model

X_sample = [w.replace(' : //', '') for w in X_sample]
X_sample = [w.replace(' . ', '') for w in X_sample]
X_sample = [w.replace('/', '') for w in X_sample]
X_dev_sample=[w.replace(' : //', '')  for w in X_dev_sample]
X_dev_sample=[w.replace(' . ', '') for w in X_dev_sample]
X_dev_sample=[w.replace('/', '') for w in X_dev_sample]

cv4 = CountVectorizer(ngram_range=(1,2),stop_words='english')
X_train=cv4.fit_transform(X_sample)
X_dev=cv4.transform(X_dev_sample)
clf_4 = MultinomialNB(alpha=1,fit_prior=False)
clf_4.fit(X_train, y_train) 
y_predicted4=clf_4.predict(X_dev)
print ("Best trained model: Multinomial Naive Bayes with combination of unigram and bigrams")
print ("Accuracy is %.5f" 
       % (sklearn.metrics.accuracy_score(y_dev, y_predicted4)))

features4=cv4.get_feature_names()
coefficient4=clf_4.coef_[0]
print ('Top 20 features are:')
print (sorted(zip(coefficient4, features4), reverse=True)[:20])
cm4 = confusion_matrix(y_dev, y_predicted4)
print ('Confusion Matrix:')
print(cm4)