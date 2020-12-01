
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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from scipy.special import comb
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from itertools import combinations 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split, cross_val_score


#load dataframe with classifications

#this is just an example of what you should output AFTER classification using both shuffling methods + model types 
df = pd.DataFrame(columns=['trial','X', 'X2', 'Movement', 'Y', 'Classification_US_l', 'Classification_US_nl', 'Classification_S_l','Classification_S_nl' ])

#this is juts sample data 
for i in range(50):
    df.loc[i]=[str(i)] +[np.random.randint(0,2) for n in range(8)]
    
 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score, precision_recall_curve, auc,plot_precision_recall_curve


def BootstrapCoef(data,numboot):
    clf = LogisticRegression(penalty='none', max_iter=1000, solver='lbfgs')
    numboot = 1000
    n = len(data)
    accuracy = np.zeros((numboot, 1))    
    for i in range(numboot):
        d = data.sample(n, replace=True)
        X_fit = np.c_[d.X.values.astype(int), d.X2.values.astype(int)]
        cver = StratifiedKFold(n_splits = 5)
        accuracy = cross_val_score(clf, X_fit,d.Y.astype(int), cv=cver, scoring = 'accuracy')
    return accuracy

params = BootstrapCoef(df,100)
#do bootstrapping
#make anova dataframe
