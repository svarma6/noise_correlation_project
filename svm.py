
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer


#for all data regardless of active or passive 
#or one for active one for passive

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

#Define features X: firing rate data

#Define predicted class: 4 (or 8) target locations

#train test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'C':[1, 10]}
svc = svm.SVC(decision_function_shape={‘ovo’}))
clf = make_pipeline(StandardScaler(), GridSearchCV(svc, parameters)) #Scale the firing rate to prevent large values from dominating the classfication,..
#...and search for best kernal and regularization values

#5-fold stratified cross validation to assess the classifier's performance
cver = StratifiedKFold(n_splits = 5)
scores = cross_val_score(clf, X, y, cv=cver, scoring = 'accuracy')
