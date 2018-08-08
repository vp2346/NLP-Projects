import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys


import sys

a = sys.argv[1]  #train.txt 
b = sys.argv[2]  #test.txt

def read_extract(filename):
    #with open(filename,"r",encoding="ascii") as f:   
    with open(filename,"r") as f:
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

X_sample=read_extract(a)[0]
y_train=read_extract(a)[1]
X_dev_sample=read_extract(b)[0]
y_dev=read_extract(b)[1]


#MB(1,2) model

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
print ("Accuracy for best trained model is %.5f"
       % sklearn.metrics.accuracy_score(y_dev, y_predicted4))

pickle.dump(clf_4, open('model.pkl', 'wb'))
pickle.dump((X_dev, y_dev), open("test.pkl", 'wb'))
pickle.dump(cv4, open('features.pkl','wb'))