#Author: Sonia Yasmin
#email: soniayasmin1995@gmail.com

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


#what  df should look like after classification 
#Y column should be target
#Classification_US should be the y_pred for the unshuffled model
#Classification_S should be the y_pred for the shuffled model

df = pd.DataFrame(columns=['trial','X', 'X2', 'Movement', 'Y', 'Classification_US', 'Classification_S' ])

#this is juts sample data 
for i in range(50):
    df.loc[i]=[str(i)] +[np.random.randint(0,2) for n in range(6)]
    
    
#confusion matrices for both
c_mat_US=confusion_matrix(df.Y.values.astype(int), df.Classification_US.values.astype(int))
c_mat_S=confusion_matrix(df.Y.values.astype(int), df.Classification_S.values.astype(int))


#change 'Classification_US', 'Classification_S'  to if the prediction was correc or not
df.Classification_US= (df.Classification_US.values==df.Y.values).astype(int)
df.Classification_S= (df.Classification_S.values==df.Y.values).astype(int)

print(df)

#contingency table 
trials=df.Classification_US.values.shape[0]

yUS_yS= df.loc[(df['Classification_US'] == 1) & (df['Classification_S'] == 1)]
nUS_yS= df.loc[(df['Classification_US'] == 0) & (df['Classification_S'] == 1)]
yUS_nS= df.loc[(df['Classification_US'] == 1) & (df['Classification_S'] == 0)]
nUS_nS= df.loc[(df['Classification_US'] == 0) & (df['Classification_S'] == 0)]

contingincey_table= [[len(yUS_yS), len(yUS_nS)],
                     [len(nUS_yS),len(nUS_nS) ]]

#Mcnmar's test
result = mcnemar(contingincey_table, exact=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')
    
      
#Compare values specificically for passive and active 

data_passive=df.loc[df['Movement'] == 0]
data_active=df.loc[df['Movement'] == 1]


active_trials=data_active.Classification_US.values.shape[0]
passive_trials=data_passive.Classification_US.values.shape[0]

frequency_table= [[np.sum(data_passive.Classification_US.values),
                          np.sum(data_passive.Classification_S.values),
                          np.sum(data_active.Classification_US.values),
                          np.sum(data_active.Classification_S.values)],
                         [passive_trials-(np.sum(data_passive.Classification_US.values)),
                          passive_trials-(np.sum(data_passive.Classification_S.values)),
                          active_trials-(np.sum(data_active.Classification_US.values)), 
                          active_trials-(np.sum(data_active.Classification_S.values))]]
  
val, p, dof, expected= chi2_contingency(frequency_table)

#which values are different than expected 
frequency_table- expected

#post-hoc test? 

#Bootstrapping 

def BootstrapCoef(data,numboot, numfeatures):
    model = svm.SVC() #whatever classifier we end up making 
    numboot = 1000
    n = len(data)
    accuracy = np.zeros((numboot, 1))    
    for i in range(numboot):
        d = data.sample(n, replace=True)
        X_fit = np.c_[d.X1, d.X2] #add total number of features 
        cver = StratifiedKFold(n_splits = 5)
        accuracy = cross_val_score(clf, X_fit,d.Y, cv=cver, scoring = 'accuracy')
    return accuracy

unshuffled_errors = BootstrapCoef(df,numboot, numfeatures)
shuffled_errors = BootstrapCoef(df,numboot, numfeatures)

bs_us_res = unshuffled_errors- clf.fit(X_fit,d.Y).score(X_fit,d.Y)
bs_s_res = shuffled_errors- clf2.fit(X_fit,d.Y).score(X_fit,d.Y)

plt.hist(bs_us_res, edgecolor = 'white', density=True)

us_ci_lower, us_ci_upper = np.quantile(bs_us_res, [0.025, 0.975])
us_boot_ci = [clf.fit(X_fit,d.Y).score(X_fit,d.Y) - us_ci_upper, 
           clf.fit(X_fit,d.Y).score(X_fit,d.Y) - us_ci_lower]

s_ci_lower, s_ci_upper = np.quantile(bs_s_res, [0.025, 0.975])
s_boot_ci = [clf.fit(X_fit,d.Y).score(X_fit,d.Y) - s_ci_upper, 
           clf.fit(X_fit,d.Y).score(X_fit,d.Y) - s_ci_lower]
