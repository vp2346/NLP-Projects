

import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import sys

a = sys.argv[1]  #model.pkl
b = sys.argv[2]  #test

trained_model = pickle.load(open(a, 'rb'))
b = pickle.load(open('test.pkl', 'rb'))
X_test=b[0]
y_test=b[1]


featuresCV=pickle.load(open('features.pkl','rb'))
features=featuresCV.get_feature_names()
coefficient=trained_model.coef_[0]
print ('Top 20 features for best trained model are:')
print (sorted(zip(coefficient, features), reverse=True)[:20])
from sklearn.metrics import confusion_matrix
y_pred=trained_model.predict(b[0])
cm = confusion_matrix(b[1], y_pred)
print ('Confusion Matrix for best trained model:')
print(cm)
import matplotlib.pyplot as plt
plt.figure()
classes=['democrat','republican']
plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0,1]
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()