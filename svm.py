
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
parameters = {'kernel':('linear', 'rbf', 'guassian'), 'C':[1, 10]}
svc = svm.SVC(decision_function_shape{‘ovo’})
clf = GridSearchCV(svc, parameters)
