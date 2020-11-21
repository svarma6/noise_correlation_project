#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:42:51 2020

@author: Dimitri

loads data from MATLAB data struct files, as provided by the creators of the dataset
saves the data from each recording session as a Pandas data frame, with trials as rows.


"""

import scipy.io as sio
import numpy as np
import pandas as pd


def makeTable(dat):
    nTrials    = len(dat)
    tgtDir     = np.array(dat['tgtDir'])
    bumpDir    = np.array(dat['bumpDir'])    
    passive    = np.array(dat['ctrHoldBump'])
    
    # make single variable for dir
    direction  = np.zeros(nTrials)
    direction[passive==1] = bumpDir[passive==1]
    direction[passive==0] = tgtDir[passive==0]
    
    # other variables
    startIndex = np.array(dat['idx_movement_on'])
    unitGuide  = np.array(dat['S1_unit_guide'])
    spikes     = np.array(dat['S1_spikes'])
    firingRate = np.array(dat['S1_firingRate'])
    df = pd.DataFrame({'passive':passive,
                       'direction':direction,
                       'startIndex':startIndex,
                       'unitGuide':unitGuide,
                       'spikes':spikes,
                       'firingRate':firingRate})
    
    # categorical direction variables
    df["dir_cat"] = df["direction"].apply(str).astype('category')
    # make variable with categories: direction separate for act vs pas
    directions_x_passive = ["%.0f" % number for number in direction]
    for i in range(nTrials):
        if passive[i] == 1:
            directions_x_passive[i] = directions_x_passive[i] + '_passive'
    df['directions_x_passive'] = directions_x_passive
    df['directions_x_passive'] = df['directions_x_passive'].astype('category')
    # make variable which separate categories for direction x active/passive
    #directionxpassive = direction + passive*np.unique(direction).shape[0]
    
    return df

def main():
    # directory of .mat data files
    dataDir = "/Users/Dimitri/Dropbox (SensorimotorSuperlab)/project9637/Data/preprocessed/"
    #names of files to load
    file_names = ["C_20170912_COactpas_TD.mat",
                  'C_20170913_COactpas_TD.mat',
                  'H_20171204_COactpas_TD.mat',
                  'H_20171207_COactpas_TD.mat']
    #names of files to save
    save_file_names = ['C_1.pkl',
                  'C_2.pkl',
                  'H_1.pkl',
                  'H_2.pkl']

    # for each file
    for filei in range(len(file_names)):
    #for filei in range(2):
        print('loading file: ',dataDir + file_names[filei])
        # load the file
        mat_contents = sio.loadmat(dataDir + file_names[filei],squeeze_me=True)
        dat = mat_contents['td_trim']
        # convert to pd
        pd = makeTable(dat)
        print('saving file: ',dataDir + save_file_names[filei])
        pd.to_pickle(dataDir + save_file_names[filei])
        
if __name__ == "__main__":
    main()