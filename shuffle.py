#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:54:26 2020

@author: Dimitri
"""

import pandas as pd
import numpy as np

def shuffle_noise_correlation(df,response):
    # function to remove noise correlation in neural data
    # inputs:
    #    -df: dataframe with columns for each neuron + categorical condition/stimulus
    #    -response: string specifying the name of the column coding condition/stimulus
    # method: 
    #    -for each neuron, randomly shuffles trials within each condition
    #    -this preserves the average neuronal responses within conditions
    #    -any correlated noise/variability across neurons within conditions is destroyed
    # 
    response_vals = df[response]
    cats = np.unique(response_vals)
    predictors = df.drop(response,axis=1)
    x = predictors.to_numpy()
    for categ in cats:
        categ_idx = np.where(response_vals == categ)[0]
        for predictori in range(predictors.shape[1]):
            x[categ_idx,predictori] = x[np.random.permutation(categ_idx),predictori]
    df_shuffle = pd.DataFrame(x) 
    df_shuffle[response] =  response_vals    
    return df_shuffle

def main():
    
    # directory of .mat data files
    dataDir = "/Users/Dimitri/Dropbox (SensorimotorSuperlab)/project9637/Data/preprocessed/"
    #names of files to load
    file_names = ['C_1',
                  'C_2',
                  'H_1',
                  'H_2']
    filei = 0
    df = pd.read_pickle(dataDir + file_names[filei] + '_binned' + '.pkl')
    df = df.drop('passive',axis=1)
    df = df.drop('direction',axis=1)
    df_shuffle = shuffle_noise_correlation(df,'directions_x_passive')

