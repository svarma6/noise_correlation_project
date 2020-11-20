
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
parameters = {'kernel':('linear', 'rbf', 'guassian'), 'C':[1, 10]}
svc = svm.SVC(decision_function_shape{‘ovo’})
clf = GridSearchCV(svc, parameters)

#for all data regardless of active or passive 
#or one for active one for passive

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
parameters = {'kernel':('linear', 'rbf', 'guassian'), 'C':[1, 10]}
svc = svm.SVC(decision_function_shape={‘ovo’}))
clf = GridSearchCV(svc, parameters)

#Define features X: firing rate data

#Define predicted class: 4 (or 8) target locations

#Scale the firing rate to prevent large values from dominating the classfication

#5-fold cross validation to assess the classifier's performance
scores = cross_val_score(clf, X, y, cv=5)
