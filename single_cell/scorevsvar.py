#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:23:52 2019

@author: kai

v3: individual planes
v2: restructured folders
"""

#from pyglmnet import GLM
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from dir_tuning_alllayers_mp import *
import os

# %% PARS AND FIRST OVERVIEW

# GLOBAL PARS
#modelname = 'spatial_temporal_4_8-16-16-32_64-64-64-64_5272'
#modelname = 'temporal_spatial_4_16-16-32-64_64-64-64-64_5272'
#modelname = 'spatiotemporal_4_8-8-32-64_7292'
#t_kernelsize = 7
t_stride = 2
ntime=320
metrics = ['RMSE', 'r2', 'PCC']
nmetrics = len(metrics)

# %% VAR VS SCORE

def scorevsvarplot(modevals):
    
    modnames = ['Dir', 'Vel', 'Dir x Vel', 'Acc', 'Labels']
    nmods = len(modnames)
        

    fig = plt.figure(figsize=(14,6), dpi=200)
    #[at]

    for i in range(nmods):
        
        plt.scatter(modevals[5], modevals[i])
        
    plt.title('Score vs. Var for the Different Tuning Features')
    
    plt.xlabel('Variance')
    plt.ylabel('r2 Scores')
    plt.ylim(-0.1, 1.1)
    #plt.xlim(0, nlayers * lspace + 1)

    plt.legend(modnames)
    return fig

def scorevsvar(model, runinfo):
    
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    #modeltype = model['type']
    
    #resultsfolder = runinfo.resultsfolder(model)
    ff = runinfo.analysisfolder(model, 'scorevsvar')
    
    expf={
          'vel': runinfo.resultsfolder(model, 'vel'),
          'acc': runinfo.resultsfolder(model, 'acc'),
          'labels': runinfo.resultsfolder(model, 'labels')
          }
    
    lmevals = []
    
    #READ IN DATA
    for ilayer in np.arange(0,nlayers):
        lmevals.append([])
        
        dvevals = np.load(os.path.join(expf['vel'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'vel', 'std', runinfo.planestring())))
        accevals = np.load(os.path.join(expf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std',runinfo.planestring())))
        labevals = np.load(os.path.join(expf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
        
        lmevals[ilayer].append(dvevals[...,1,1]) #dir
        lmevals[ilayer].append(dvevals[...,2,1]) #vel
        lmevals[ilayer].append(dvevals[...,3,1]) #dir + vel
        lmevals[ilayer].append(accevals[...,2,1]) #acc
        lmevals[ilayer].append(labevals[...,21,1]) #labels
        lmevals[ilayer].append(labevals[...,20,1]) #variance
        
    # PLOT
    os.makedirs(ff, exist_ok = True)
    
    for ilayer in np.arange(0, nlayers):
        fig = scorevsvarplot(lmevals[ilayer])
        fig.savefig(os.path.join(ff, 'scorevsvar_l%d_%s_test.png' %(ilayer, runinfo.planestring())))
        plt.close('all')
        print('plot saved')