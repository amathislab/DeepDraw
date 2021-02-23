#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 2021ka

@author: kai

This performs the CKA between the models trained on regression task and the models trained on the character recognition task
"""

import numpy as np
import pandas as pd
from representational_similarity_analysis import models_cka_matrix

from rowwise_neuron_curves_controls import lstring, read_layer_reps
import os, pickle
import copy
import matplotlib.pyplot as plt

def main(task_models, regression_models, runinfo, modelinfo):

    regressiontaskfolder = runinfo.regressiontaskfolder(modelinfo)
    df_path = os.path.join(regressiontaskfolder, 'cka_df.csv')
    max_nlayers = modelinfo['nlayers'] #assumption is the base model in modelinfo contains maximum amount of layers (case for all model types seen so far)

    if not os.exists(df_path):
        ### compute matrix of cka scores

        #initialize dataframe
        index = task_models
        ilayers = range(max_nlayers)

        columns = pd.MultiIndex.from_tuples(
            [('nlayers', 'nlayer'),] + 
            [('cka_scores', ilayer) for ilayer in ilayers)

        df = pd.DataFrame(index=index, columns=columns)

        #compute cka scores
        for task_model, regression_model in zip(task_models, regression_models):

            #initialize model info dicts
            task_model_info = modelinfo.copy()
            task_model_info['base'] = task_model_info['name'] = task_model
            
            regression_model_info = modelinfo.copy()
            regression_model_info['base'] = regression_model_info['name'] = regression_model
            
            #compute nlayers
            task_model_datafolder = runinfo.datafolder(task_model_info)
            task_model_datafiles = list(next(os.walk(task_model_datafiles))[2])
            nlayers = len(task_model_datafiles)

            task_model_info['nlayers'] = regression_model_info['nlayers'] = nlayers

            #set corresponding row of dataframe
            df[('nlayers', 'nlayer')] = nlayers
            df['cka_scores'].loc[task_model][:nlayers] = models_cka_matrix(task_model_info, regression_model_info, runinfo)

        df.to_csv(df_path)

    else:

        df = pd.read_csv(df_path)

    ##make plots
    print('making regression task plots...')
    for ilayer in range(max_nlayers):
        print('making histogram for layer %d ...' %ilayer)
        layer_cka_scores = df[('cka_scores', ilayer)]

        plt.hist(layer_cka_scores.flatten())
        plt.savefig(os.path.join(regressiontaskfolder, 'histogram_layer%d.png' %ilayer)
    