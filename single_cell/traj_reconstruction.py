#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:23:11 2019

@author: kai
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from dir_tuning_alllayers_mp import *
from rowwise_neuron_curves_controls import *
import os
from sklearn.linear_model import Ridge

decfsets = ['ee', 'vel']
mmod = 'decoding'
ntime = 320

# %% PREFERRED DIRECTION GENERALIZATION

def rec_traj_plot(model, runinfo, orig_traj, rec_traj):
    
    fig = plt.figure(dpi = 250)
    plt.plot(orig_traj[0], orig_traj[1])
    plt.plot(rec_traj[0], rec_traj[1], 'r--')
    plt.title('Recovered Trajectories')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.legend(['original', 'decoded'])
    return fig

def reconstruct_trajectories(model, runinfo, fset):
    kin, xyplmvt = X_data(fset=fset, runinfo=runinfo, datafolder = runinfo.datafolder(model))
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    
    resultsfolder = runinfo.resultsfolder(model, 'decoding_%s' %fset)
    analysisfolder = runinfo.analysisfolder(model, 'decoding_%s_traj'%fset)
    
    for i, ridx in enumerate(np.random.permutation(int(len(kin)*ntime))[:10]):
        for ilayer in np.arange(-1, nlayers-1):
            layer = lstring(ilayer)
            lo = pickle.load(open(os.path.join(runinfo.datafolder(model), layer + '.pkl'), 'rb'))
            lo = lo[xyplmvt]
            
            Y = lo
            X = kin
            
            print(X.shape)
            print(Y.shape)
            
            #centers = get_centers(lo.shape[2])
            
            Y = Y.swapaxes(1,2).reshape((Y.shape[0], Y.shape[2], -1)).swapaxes(1,2)
        
            """
            if(len(X.shape) > 1):
                x = X[..., centers]
            else:
                x = X
            """
            
            #x = X[:,ikin]
            x = X
            y = Y
            
            ##RESHAPE FOR LINEAR REGRESSION
            #print(x.shape, y.shape)
            """
            if len(x.shape) > 1: ##FOR TIME-BASED DATA ( 2 COMPS PER TIMESPACE IS USE CASE)
                tcoff = sum(np.where(centers <= 32, True, False))
                x = x[...,tcoff:ntime-tcoff]
                y = y[...,tcoff:ntime-tcoff]
            
            elif fset == 'labels':
                temp = np.ones_like(y)
                x = (temp.swapaxes(0,1) * x).swapaxes(0,1)
                x = x.reshape((-1,))
            """    
            #print(x.shape, y.shape)
            
            #x = x.reshape((-1))
            #x = x.swapaxes(1,2).reshape((-1,2))
            #y = y.swapaxes(1,2).reshape((-1,y.shape[1]))
            x = x.swapaxes(1,2).reshape((-1,2))
    #print(x.shape, y.shape)

            y = y.swapaxes(1,2).reshape((-1,y.shape[1]*y.shape[2]))
            
            #X_train, X_test, Y_train, Y_test = train_test_split(y, x, test_size=0.20, random_state=runinfo.randomseed_traintest)

            #print(X_test.shape, Y_test.shape)
            
            orig_traj = x[ridx].reshape((2,-1))
            
            with open(os.path.join(resultsfolder, 
                                   'l%d_%s_mets_%s_%s_ridgemodel.pkl' 
                                   %(ilayer + 1, fset, 'decoding', runinfo.planestring())),
                    'rb') as file:
                lm = pickle.load(file)
            
            rec_traj = lm.predict(y[int(ridx/ntime)].reshape(1,-1)).reshape((2,-1))
            
            fig = rec_traj_plot(model, runinfo, orig_traj, rec_traj)
            
            os.makedirs(os.path.join(analysisfolder, 'traj%d' %i), exist_ok=True)
            
            fig.savefig(os.path.join(analysisfolder, 'traj%d' %i, 'reconstruction_l%d_%d.png' %(ilayer + 1, i)))
            plt.close('all')
    
def rec_scores_plot(model, runinfo, scores):
    nlayers = model['nlayers'] + 1
    fig = plt.figure(dpi=250)
    plt.plot(np.arange(nlayers), scores, marker='D')
    plt.title('Reconstruction Accuracy')
    plt.ylabel('r2 Scores')
    plt.xlabel('Layer')
    plt.ylim(-0.1, 1.1)

    return fig
    
def reconstruction_scores(model, runinfo, fset):
    modelname = model['name']    
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    base = model['base']
        
    resultsfolder = runinfo.resultsfolder(model, 'decoding_%s' %fset)
    analysisfolder = runinfo.analysisfolder(model, 'decoding_%s_scores'%fset)
    
    # READ IN ALL EVALS
    alltestevals = []
    alltrainevals = []
        
    for ilayer in np.arange(0,nlayers):
        testevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(resultsfolder, ilayer, fset, mmod, runinfo.planestring()))        
        alltestevals.append(testevals)
        
        trainevals = np.load('%s/l%d_%s_mets_%s_%s_train.npy' %(resultsfolder, ilayer, fset, mmod, runinfo.planestring()))        
        alltrainevals.append(trainevals)
    
    #COMPUTE HISTOGRAMS
    os.makedirs(analysisfolder,exist_ok=True )
    
    print(alltestevals)
    r2 = [testevals[0,1] for testevals in alltestevals]
    
    fig = rec_scores_plot(model, runinfo, r2)
    fig.savefig(os.path.join(analysisfolder, '%s_dec_scores_l%d.png' %(fset, ilayer)))
    plt.close('all')
    
# %% GENERALIZATION MAIN
    
def main(model, runinfo):
    print('outputting score plots...')
    for fset in decfsets:
        #if(not os.path.exists(runinfo.analysisfolder('%s_decoding_scores' %fset))):
        if(True):
            reconstruction_scores(model, runinfo, fset)
    
    print('reconstructing trajectories...')
    #for fset in decfsets:
        #if(not os.path.exists(runinfo.analysisfolder('%s_decoding' %fset))):
    fset = 'ee'
    if(True):
        reconstruct_trajectories(model, runinfo, fset)
