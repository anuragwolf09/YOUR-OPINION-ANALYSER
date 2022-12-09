#SUPPORT VECTOR CLASSIFIER:

# from sklearn.svm import SVC
svc=SVC()
svc.fit(train_data,y_train)
svc.score(test_data,y_test)
#ouput:0.7169177574709805
#therefore we get 71% accuracy using Support vector classifier.

#RANDOM FOREST:

from sklearn.ensemble import RandomForestClassifier as rf
random_forest=rf()
random_forest.fit(train_data,y_train)
random_forest.score(test_data,y_test)
#output:0.9138058779945666
#therefore we get decent 91% score with random forest.

#we can also look at classification report for random forest:
from sklearn.metrics import classification_report as cr
print(cr(y_test,y))
"""
             precision    recall  f1-score   support

   Negative       0.90      0.90      0.90      8521
   Positive       0.93      0.92      0.93     11724

avg / total       0.91      0.91      0.91     20245
"""

#MULTINOMIAL NAIVE BAYES:

from sklearn.naive_bayes import MultinomialNB as mn
naive_bayes=mn()
naive_bayes.fit(train_data,y_train)
naive_bayes.score(test_data,y_test)
#OUPUT:0.8034082489503581
#therefore we get 80% score through multinomial naive bayes

#CONCLUSION:
# we can see that we are getting very nice accuracy by random forest.


#now we will try with kaggle test data. Head to classifying_through_test_data.py