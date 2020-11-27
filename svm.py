#Author: Sonia Yasmin 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from scipy.special import comb
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

#for all data regardless of active or passive 
#or one for active one for passive

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

#Read dataframes for each both Monkey C and H and their associated trials 
df_C1 = pd.read_pickle("C_1_binned.pkl")
df_C2 = pd.read_pickle("C_2_binned.pkl")
df_H1 = pd.read_pickle("H_1_binned.pkl")
df_H2 = pd.read_pickle("H_2_binned.pkl")

#Define predicted classes: drection_X_passive. 8 classes ['0', '0_passive', '180', '180_passive', '270', '270_passive', '90', '90_passive']
y_C1 = df_C1["directions_x_passive"]
y_C2 = df_C2["directions_x_passive"]
#Define predicted classes: drection_X_passive. 16 classes ['0', '0_passive', '135', '135_passive', ..., '45', '45_passive', '90', '90_passive']
y_H1 = df_H1["directions_x_passive"]
y_H2 = df_H2["directions_x_passive"]

#Drop columns and define features X: firing rate data
X_C1 = df_C1.drop(["direction","passive","directions_x_passive"], axis = 1)
X_C2 = df_C2.drop(["direction","passive","directions_x_passive"], axis = 1)
X_H1 = df_H1.drop(["direction","passive","directions_x_passive"], axis = 1)
X_H2 = df_H2.drop(["direction","passive","directions_x_passive"], axis = 1)

#Below are codes for Monkey C trial #1 from "C_1_binned.pkl" only

#train test split
Xtrain_C1, Xtest_C1, ytrain_C1, ytest_C1 = train_test_split(X_C1, y_C1, test_size=0.5, random_state=0)

#Set the parameters for tuning
parameters = {'kernel':('linear', 'rbf', 'sigmoid','poly'), 
              'C':[0.1, 1, 10, 100, 1000],
              'gamma':[1,0.1,0.001,0.0001,'scale','auto']}
svc = SVC(decision_function_shape='ovo')
#Standardize values and search for best kernal and parameters with 10-fold cross validation
clf = make_pipeline(StandardScaler(), 
                    GridSearchCV(svc, parameters,cv=10,refit=True))
clf.fit(Xtrain_C1, ytrain_C1)

#Check the optimal parameters
print(clf[1].best_params_) #which are {'C': 10, 'gamma': 0.001, 'kernel': 'sigmoid'}

#Assess the model performance on test dataset
#Classification_US is the y_pred for the unshuffled model
Classification_US_C1 = clf.predct(Xtest_C1)
print(classification_report(ytest_C1,Classification_US_C1))

