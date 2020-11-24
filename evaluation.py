import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency


#what  df should look like after classification 
#Y column should be target
#Classification_US should be the y_pred for the unshuffled model
#Classification_S should be the y_pred for the shuffled model

df = pd.DataFrame(columns=['trial', 'Movement', 'Y', 'Classification_US', 'Classification_S' ])

#this is juts sample data 
for i in range(50):
    df.loc[i]=[str(i)] +[np.random.randint(0,2) for n in range(4)]
    
    
#confusion matrices for both
c_mat_US=confusion_matrix(df.Y.values.astype(int), df.Classification_US.values.astype(int))
c_mat_S=confusion_matrix(df.Y.values.astype(int), df.Classification_S.values.astype(int))


#change 'Classification_US', 'Classification_S'  to if the prediction was correc or not
df.Classification_US= (df.Classification_US.values==df.Y.values).astype(int)
df.Classification_S= (df.Classification_S.values==df.Y.values).astype(int)

print(df)

#contingency table 
trials=df.Classification_US.values.shape[0]

contingincey_table= [[np.sum(df.Classification_US.values), np.sum(df.Classification_S.values)],
                     [trials-(np.sum(df.Classification_US.values)), trials-(np.sum(df.Classification_S.values))]]

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

big_contingincey_table= [[np.sum(data_passive.Classification_US.values),
                          np.sum(data_passive.Classification_S.values),
                          np.sum(data_active.Classification_US.values),
                          np.sum(data_active.Classification_S.values)],
                         [passive_trials-(np.sum(data_passive.Classification_US.values)),
                          passive_trials-(np.sum(data_passive.Classification_S.values)),
                          active_trials-(np.sum(data_active.Classification_US.values)), 
                          active_trials-(np.sum(data_active.Classification_S.values))]]
  
val, p, dof, expected= chi2_contingency(big_contingincey_table)

#which values are different than expected 
big_contingincey_table- expected
