#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:56:22 2020

@author: kai
"""

import numpy as np
import os

import pandas as pd

# %% EXTRACT LABEL SCORES

filefolder = 'exp104/results/spatial_temporal_4_8-16-16-32_64-64-64-64_5272/spatial_temporal_4_8-16-16-32_64-64-64-64_5272_1/horall/labels/'
filename = 'l8_labels_mets_std_horall_test.npy'

labeltuning = np.load(os.path.join(filefolder, filename))

# %% VEL TUNING

#model = 'spatiotemporal_4_8-8-32-64_7272'
model = 'temporal_spatial_4_16-16-32-64_64-64-64-64_5272'
#model = 'spatial_temporal_4_8-16-16-32_64-64-64-64_5272'
fset = 'vel'

filefolder = 'exp102/results/%s/%s_1/horall/%s/' %(model, model, fset)
filename = 'l0_%s_mets_std_horall_test.npy' %fset

veltuning = np.load(os.path.join(filefolder, filename))

print(filename)

print(np.median(veltuning[:,:,1], axis=0))
print(np.quantile(veltuning[:,:,1], 0.90, axis=0))

filename = 'l1_%s_mets_std_horall_test.npy' %fset

veltuning = np.load(os.path.join(filefolder, filename))

print(filename)

print(veltuning.shape)

print(np.median(veltuning[...,1], axis=(0,1)))
print(np.quantile(veltuning[...,1], 0.90, axis=(0,1)))

filename = 'l2_%s_mets_std_horall_test.npy' %fset

veltuning = np.load(os.path.join(filefolder, filename))

print(filename)

print(veltuning.shape)

print(np.median(veltuning[...,1], axis=(0,1)))
print(np.quantile(veltuning[...,1], 0.90, axis=(0,1)))


filename = 'l4_%s_mets_std_horall_test.npy' %fset

veltuning = np.load(os.path.join(filefolder, filename))

print(filename)

print(veltuning.shape)

print(np.median(veltuning[...,1], axis=(0,1)))
print(np.quantile(veltuning[...,1], 0.90, axis=(0,1)))


filename = 'l5_%s_mets_std_horall_test.npy' %fset

veltuning = np.load(os.path.join(filefolder, filename))

print(filename)

print(veltuning.shape)

print(np.median(veltuning[...,1], axis=(0,1)))
print(np.quantile(veltuning[...,1], 0.90, axis=(0,1)))

filename = 'l7_%s_mets_std_horall_test.npy' %fset

veltuning = np.load(os.path.join(filefolder, filename))

print(filename)

print(veltuning.shape)

print(np.median(veltuning[...,1], axis=(0,1)))
print(np.quantile(veltuning[...,1], 0.90, axis=(0,1)))


filename = 'l8_%s_mets_std_horall_test.npy' %fset

veltuning = np.load(os.path.join(filefolder, filename))

print(filename)

print(veltuning.shape)

print(np.median(veltuning[...,1], axis=(0,1)))
print(np.quantile(veltuning[...,1], 0.90, axis=(0,1)))


# %% STATS FOR CONTROL COMP

#model = 'spatiotemporal_4_8-8-32-64_7272'
#model = 'temporal_spatial_4_16-16-32-64_64-64-64-64_5272'
model = 'spatial_temporal_4_8-16-16-32_64-64-64-64_5272'

analysisfolder = 'exp102/analysis/%s/comparison/horall/kindiffs/' %model
df = pd.read_csv(os.path.join(analysisfolder, model + '_comparisons_df.csv'),
                 header=0, index_col=[0,1,2], dtype={'layer': int, 'mean': float, 'median': float})

fsets = ['dir','vel','dirvel','acc','labels']


nlayers = 8
count_total = 25 * nlayers
count_median = 0
count_q90 = 0
for run in range(5):
    run+=1
    for ilayer in range(nlayers):
        for fset in fsets:
            if(df.loc[('%s_%d'%(model, run),ilayer, fset), 'median'] > df.loc[('%s_%dr'%(model, run),ilayer, fset), 'median']):
                count_median +=1
            if(df.loc[('%s_%d'%(model, run),ilayer, fset), 'q90'] > df.loc[('%s_%dr'%(model, run),ilayer, fset), 'q90']):
                count_q90 +=1

print('Medians Trained Greater: %d / %d '%(count_median, count_total) )
print('Q90 Trained > Control: %d / %d '%(count_q90, count_total))