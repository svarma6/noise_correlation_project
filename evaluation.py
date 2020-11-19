import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

#Active Unshuffled
AU_performance= accuracy_score(ytest_au,  model.predict(Xtest_au))

#Passive Unshuffled 
PU_performance= accuracy_score(ytest_pu,  model.predict(Xtest_pu))

#Active Shuffled
AS_performance= accuracy_score(ytest_as,  model.predict(Xtest_as)) 

#Passive Shuffled 
PS_performance= accuracy_score(ytest_ps,  model.predict(Xtest_ps)) 

#dataframe=pd.DataFrame(AU_performance,PU_performance, AS_performance, PS_performance, columns=['Active Unshuffled', 'Passive Unshuffled', 'Active Shuffled', 'Passive Shuffled'] )

#2 way anova 
anova_model = ols('performance ~ C(conition) + C(shuffling) + C(condition):C(shuffling)', data=df).fit()
sm.stats.anova_lm(model, typ=2)

#Tukeys HSD
import statsmodels.stats.multicomp.pairwise_tukeyhsd
