#Author: Sonia Yasmin 
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
#Read dataframes
df_C1 = pd.read_pickle("C_1_binned.pkl")
df_C2 = pd.read_pickle("C_2_binned.pkl")
df_H1 = pd.read_pickle("H_1_binned.pkl")
df_H2 = pd.read_pickle("H_2_binned.pkl")
#Define predicted classes: drection_X_passive. 8 classes ['0', '0_passive', '180', '180_passive', '270', '270_passive', '90', '90_passive']
y_C1 = df_C1["directions_x_passive"]
y_C2 = df_C2["directions_x_passive"]
y_H1 = df_H1["directions_x_passive"]
y_H2 = df_H2["directions_x_passive"]
#Drop columns
df_C1 = df_C1.drop(["direction","passive","directions_x_passive"], axis = 1)

#train test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'C':[1, 10]}
svc = svm.SVC(decision_function_shape={‘ovo’}))
clf = make_pipeline(StandardScaler(), GridSearchCV(svc, parameters)) #Scale the firing rate to prevent large values from dominating the classfication,..
#...and search for best kernal and regularization values

#5-fold stratified cross validation to assess the classifier's performance
cver = StratifiedKFold(n_splits = 5)
scores = cross_val_score(clf, X, y, cv=cver, scoring = 'accuracy')


#Classification_US is the y_pred for the unshuffled model
Classification_US = clf.predct(Xtest)
