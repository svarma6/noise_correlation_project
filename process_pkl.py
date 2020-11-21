#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:53:07 2020

@author: Dimitri
"""
import pandas as pd
import numpy as np

def bin_firingRates(df,startOffset=0,nBins=12):
    nTr    = len(df)
    nCells = df['firingRate'][1].shape[1]
    firingRate_bin = np.zeros((nTr,nCells))
    for triali in range(nTr):
        fr = df['firingRate'][triali]
        b0 = df['startIndex'][triali] + startOffset
        b1 = b0 + nBins
        firingRate_bin[triali,:] = np.mean(fr[b0:b1,:],axis=0)

    fr_df = pd.DataFrame(firingRate_bin) 
    fr_df['passive'] = df.passive
    fr_df['direction'] = df.direction
    fr_df['directions_x_passive'] = df.directions_x_passive
    return fr_df

def main():
    # directory of .mat data files
    dataDir = "/Users/Dimitri/Dropbox (SensorimotorSuperlab)/project9637/Data/preprocessed/"
    #names of files to load
    file_names = ['C_1',
                  'C_2',
                  'H_1',
                  'H_2']
    # for each file
    for filei in range(len(file_names)):
    #for filei in range(1):
        print('loading file: ',dataDir + file_names[filei] + '.pkl')
        # load the file
        df = pd.read_pickle(dataDir + file_names[filei] + '.pkl')
        df_binned = bin_firingRates(df)
        df_binned.to_pickle(dataDir + file_names[filei] + '_binned' + '.pkl')
        
