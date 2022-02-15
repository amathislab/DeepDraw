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
from savelouts_best_controls import main as modeloutputs_main
import shutil

import gc

from utils import sorted_alphanumeric

def garbage_collect():
    print("Collecting...")
    n = gc.collect()
    print("Number of unreachable objects collected by GC:", n)
    print("Uncollectable garbage:", gc.garbage)

def main(task_models, regression_models, runinfo, modelinfo):

    regressiontaskfolder = runinfo.regressiontaskfolder(modelinfo)
    df_path = os.path.join(regressiontaskfolder, 'cka_df.csv')
    print(df_path)
    max_nlayers = modelinfo['max_nlayers']

    if not os.path.exists(df_path):
        ### compute matrix of cka scores

        #initialize dataframe
        index = task_models
        ilayers = range(max_nlayers+1)

        columns = pd.MultiIndex.from_tuples(
            [('nlayers', 'nlayers'),] + 
            [('cka_scores', str(ilayer)) for ilayer in ilayers])

        df = pd.DataFrame(index=index, columns=columns)

        base_model_path = os.path.join(runinfo.basefolder, 'models', 'ALL_%s' %modelinfo['typename'])

        #compute cka scores
        for task_model, regression_model in zip(task_models, regression_models):

            print('saving hidden layer output for %s , %s ... '%(task_model, regression_model))

            for hpsname in [task_model, regression_model]:
                
                model_to_analyse = modelinfo.copy()
                model_to_analyse['base'] = hpsname
                model_to_analyse['model_path'] = os.path.join(base_model_path, hpsname)
                model_to_analyse['path_to_config_file'] = os.path.join(base_model_path, hpsname, 'config.yaml')
                model_to_analyse['name'] = hpsname

                if hpsname in regression_models:
                    model_to_analyse['regression_task'] = True

                if not os.path.exists(runinfo.datafolder(model_to_analyse)):
                    modeloutputs_main(model_to_analyse, runinfo)

            garbage_collect()

            print('computing cka scores for %s , %s ...' %(task_model, regression_model))

            #initialize model info dicts
            task_model_info = modelinfo.copy()
            task_model_info['base'] = task_model_info['name'] = task_model
            
            regression_model_info = modelinfo.copy()
            regression_model_info['base'] = regression_model_info['name'] = regression_model
            
            #compute nlayers
            task_model_datafolder = runinfo.datafolder(task_model_info)
            
            task_model_datafiles = [f.name for f in os.scandir(task_model_datafolder) if f.is_file()]
            task_model_datafiles = sorted_alphanumeric(task_model_datafiles)

            print(task_model_datafiles)

            #task_model_datafiles = list(next(os.walk(task_model_datafiles))[2])
            #if(task_model['basetype'])
            nlayers = len(task_model_datafiles) - 3

            task_model_info['nlayers'] = regression_model_info['nlayers'] = nlayers

            #set corresponding row of dataframe
            df[('nlayers', 'nlayers')].loc[df.index == task_model] = nlayers

            print(df.head())
            df.sort_index(axis=1, inplace=True, sort_remaining=True)

            idx = pd.IndexSlice
            print(df.loc[task_model, idx['cka_scores', '0':str(nlayers)]])
            print(models_cka_matrix(task_model_info, regression_model_info, runinfo))
            df.loc[task_model, idx['cka_scores', '0':str(nlayers)]] = models_cka_matrix(task_model_info, regression_model_info, runinfo)

            #remove data from these two models
            shutil.rmtree(runinfo.datafolder(task_model_info))
            shutil.rmtree(runinfo.datafolder(regression_model_info))

            garbage_collect()

        os.makedirs(regressiontaskfolder, exist_ok=True)
        df.to_csv(df_path)

    else:

        df = pd.read_csv(df_path, header=[0,1], index_col=0)

        if modelinfo['type'] == 'S':
            df[('nlayers', 'nlayers')] = df.index.map(lambda x : int(x[17])*2) #why do i need this? bug where nlayers gets set to max every time...
            #ONLY FOR S MODELS SOMEHOW
        print(df.head())
        #print(df.columns)
        #print(df.index)

        #print('nlayers', df[('nlayers', 'nlayers')])

    ##make plots
    print('making regression task plots...')
    for nlayers in range(0, max_nlayers+1):
        print('making histogram for nlayers %d ...' %nlayers)
        for ilayer in range(nlayers+1):
            print('making histogram for ilayer %d / nlayers %d ...' %(ilayer, nlayers))
            df_nlayers = df.loc[df[('nlayers', 'nlayers')] == nlayers]
            #print(df_nlayers)

            layer_cka_scores = df_nlayers[('cka_scores', str(ilayer))].values
            print(layer_cka_scores)
            fig = plt.figure()
            plt.hist(layer_cka_scores, range=[0,1.1])
            fig.savefig(os.path.join(regressiontaskfolder, 'histogram_nlayers%d_layer%d.png' %(nlayers, ilayer)))
            fig.savefig(os.path.join(regressiontaskfolder, 'histogram_nlayers%d_layer%d.svg' %(nlayers, ilayer)))
            plt.close()