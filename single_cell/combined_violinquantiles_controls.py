#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:23:52 2019

@author: kai

"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rowwise_neuron_curves_controls import *
import os

def format_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

# %% PARS AND FIRST OVERVIEW

# GLOBAL PARS
t_stride = 2
ntime=320
metrics = ['RMSE', 'r2', 'PCC']
nmetrics = len(metrics)
expid = 100
fset = 'vel'
mmod = 'std'
tcoff = 32

modnames = ['Dir', 'Vel', 'Dir x Vel', 'Acc', 'Labels']
nmods = len(modnames)

# %% Get Model Evaluations

def get_modevals(model, runinfo):
    ''' Read in tuning curve fits from saved numpy files
    
    Arguments
    ---------
    model : dict, information about current model
    runinfo : RunInfo (extension of dict), information about experimental run
    
    Returns
    -------
    modevals : list of lists containing tuning curve test r2 strengths for five different model types to be plotted
        Outer list: layers; Inner list: Model types, containing np.array of r2 strengths
        
    '''
    
    expf={
          'vel': runinfo.resultsfolder(model, 'vel'),
          'acc': runinfo.resultsfolder(model, 'acc'),
          'labels': runinfo.resultsfolder(model, 'labels')
    }
    
    # READ IN OF REGS AND THRESHOLD, SAVE TEXT FILE
    modevals = []
    
    for i in range(nmods):
        modevals.append([])
    
    for ilayer in np.arange(0,model['nlayers'] + 1):
        
        dvevals = np.load(os.path.join(expf['vel'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'vel', mmod, runinfo.planestring())))
        accevals = np.load(os.path.join(expf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std',runinfo.planestring())))
        labevals = np.load(os.path.join(expf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
        
        modevals[0].append(dvevals[...,1,1]) #dir
        modevals[1].append(dvevals[...,2,1]) #vel
        modevals[2].append(dvevals[...,3,1]) #dir + vel
        modevals[3].append(accevals[...,2,1]) #acc
        modevals[4].append(labevals[:,0]) #labels
        
        
    return modevals

# %% COMPARISON VIOLIN PLOT 
    
def clip(vp, lr):
    
    for b in vp['bodies']:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
    
        if lr == 'l':
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        elif lr == 'r':
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
       
           
def plot_compvp(trainedmodevals, controlmodevals, trainedmodel):
    ''' Plot the comparison violin plot showing the distribution of tuning strengths
    
    Arguments
    ---------
    trainedmodevals : list [nr layers] of np.arrays, store performance of each neuron in trained model
    controlmodevals : list [nr layers] of np.arrays, store performance of each neuron in control model
    trainedmodel : dict
    
    Returns
    -------
    fig : plt.figure, comparison violin plot
    '''
       
    nlayers = trainedmodel['nlayers'] + 1
    
    space = 0.8
    width = 0.6
    lspace = space*nmods + 1
    
    trainedcmap = matplotlib.cm.get_cmap(trainedmodel['cmap']) #spatial_temporal    
    controlcmap = matplotlib.cm.get_cmap('Greys_r') #spatial_temporal    
    cmaps = [trainedcmap, controlcmap]
    
    ct = 0.6
    cidx = [i*(1-ct)/(nmods-1) for i in range(nmods)] #Blues_r option
    #plot figure
    fig = plt.figure(figsize=(14,6), dpi=200)
    ax1 = fig.add_subplot(111)
    
    plt.xticks(np.arange(lspace/2,nlayers*lspace + 1,lspace), 
               ['Sp.'] + ['L%d' %i for i in np.arange(1, nlayers+1)])
    
    ax1.set_ylabel('r2 or classification score')
    ax1.set_ylim(-0.1, 1.1)
    vps = []
    patches = []
    
    ccolorindex = 3
    for (modevals, cmap, alpha, zorder, lr) in zip([controlmodevals, trainedmodevals], [controlcmap, trainedcmap], [[0.8, 0.5], [0.8, 0.7]], [2,1], ['r', 'l']):
        for i, mod in enumerate(modevals):
                
            vp = ax1.violinplot(mod,
                positions=[ilayer*lspace+space*i+1 for ilayer in range(nlayers)], 
                showextrema = False,
                showmedians = False,
                showmeans = False,
                widths=width)
            
            for part in vp['bodies']:
                if lr == 'l':
                    part.set_facecolor(cmap(cidx[i]))
                    part.set_edgecolor(cmap(cidx[i]))
                else:
                    part.set_facecolor(cmap(cidx[ccolorindex]))
                    part.set_edgecolor(cmap(cidx[ccolorindex]))
                part.set_alpha(alpha[1])
                part.set_zorder(zorder)
    
            clip(vp, lr)
            
            patches.append(mpatches.Patch(color=cmap(cidx[i]), alpha=0.7))
            
            vps.append(vp)
            
        #Quantiles
        q = 0.9
        marker = 's'
        
        for i in [0,3]:
            mod = modevals[i]
            
            q90s = np.zeros((nlayers,))
            
            for ilayer, layer in enumerate(mod):
                q90s[ilayer] = np.quantile(layer, q)
                
            ax1.plot([ilayer*lspace+space*i+1 for ilayer in range(nlayers)], 
                      q90s, color=cmap(cidx[i]), marker=marker, alpha=alpha[0])
    
    format_axis(plt.gca())
    leg = plt.legend(patches[5:], modnames, loc='upper right')
    ax1.add_artist(leg)
    plt.legend([patches[5], patches[ccolorindex]], ['Trained', 'Control'], loc='upper right', bbox_to_anchor=(0.87, 1))
    
    return fig

def get_modevals_ee(model, runinfo):
    ''' Read in tuning curve fits for endeffector positional input from saved numpy files
    
    Arguments
    ---------
    model : dict, information about current model
    runinfo : RunInfo (extension of dict), information about experimental run
    
    Returns
    -------
    modevals : list of lists containing tuning curve test r2 strengths for five different model types to be plotted
        Outer list: layers; Inner list: Model types, containing np.array of r2 strengths
        
    '''
    
    expf={
          'ee': runinfo.resultsfolder(model, 'ee'),
    }
    
    # READ IN OF REGS AND THRESHOLD, SAVE TEXT FILE
    modevals = []
    nmods_ee = 2
    
    for i in range(nmods_ee):
        modevals.append([])
    
    for ilayer in np.arange(0,model['nlayers'] + 1):
        
        eeevals = np.load(os.path.join(expf['ee'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'ee', mmod, runinfo.planestring())))
        
        modevals[0].append(eeevals[...,0,1])
        modevals[1].append(eeevals[...,3,1])
        
    return modevals


def plot_compvp_ee(trainedmodevals, controlmodevals, trainedmodel):
    ''' Plot the comparison violin plot showing the distribution of tuning strengths
    
    Arguments
    ---------
    trainedmodevals : list [nr layers] of np.arrays, store performance of each neuron in trained model
    controlmodevals : list [nr layers] of np.arrays, store performance of each neuron in control model
    trainedmodel : dict
    
    Returns
    -------
    fig : plt.figure, comparison violin plot
    '''
    
    nlayers = trainedmodel['nlayers'] + 1
    
    nmods_ee = 2
    modnames_ee = ['Cartesian', 'Polar']
    
    space = 0.8
    width = 0.6
    lspace = space*nmods_ee + 1
    
    trainedcmap = matplotlib.cm.get_cmap(trainedmodel['cmap']) #spatial_temporal    
    controlcmap = matplotlib.cm.get_cmap('Greys_r') #spatial_temporal    
    cmaps = [trainedcmap, controlcmap]
    
    ct = 0.6
    cidx = [i*(1-ct)/(nmods_ee-1) for i in range(nmods_ee)] #Blues_r option
        
    #plot figure
    fig = plt.figure(figsize=(14,6), dpi=200)
    ax1 = fig.add_subplot(111)
    
    
    plt.xticks(np.arange(lspace/2,nlayers*lspace + 1,lspace), 
               ['Sp.'] + ['L%d' %i for i in np.arange(1, nlayers+1)])
    
    ax1.set_ylabel('r2 or classification score')
    ax1.set_ylim(-0.1, 1.1)
    vps = []
    patches = []
    
    ccolorindex = 1
    for (modevals, cmap, alpha, zorder, lr) in zip([controlmodevals, trainedmodevals], [controlcmap, trainedcmap], [[0.8, 0.5], [0.8, 0.7]], [2,1], ['r', 'l']):
        for i, mod in enumerate(modevals):
                
            vp = ax1.violinplot(mod,
                positions=[ilayer*lspace+space*i+1 for ilayer in range(nlayers)], 
                showextrema = False,
                showmedians = False,
                showmeans = False,
                widths=width)
            
            for part in vp['bodies']:
                if lr == 'l':
                    part.set_facecolor(cmap(cidx[i]))
                    part.set_edgecolor(cmap(cidx[i]))
                else:
                    part.set_facecolor(cmap(cidx[ccolorindex]))
                    part.set_edgecolor(cmap(cidx[ccolorindex]))
                part.set_alpha(alpha[1])
                part.set_zorder(zorder)
    
            clip(vp, lr)
            
            patches.append(mpatches.Patch(color=cmap(cidx[i]), alpha=0.7))
            
            vps.append(vp)
            
        #Quantiles
        q = 0.9
        marker = 's'
        
        for i in [0,1]:
            mod = modevals[i]
            
            q90s = np.zeros((nlayers,))
            
            for ilayer, layer in enumerate(mod):
                q90s[ilayer] = np.quantile(layer, q)
                        
            ax1.plot([ilayer*lspace+space*i+1 for ilayer in range(nlayers)], 
                      q90s, color=cmap(cidx[i]), marker=marker, alpha=alpha[0])
    
    format_axis(plt.gca())
    leg = plt.legend(patches[2:], modnames_ee, loc='upper right')
    ax1.add_artist(leg)
    plt.legend([patches[2], patches[ccolorindex]], ['Trained', 'Control'], loc='upper right', bbox_to_anchor=(0.87, 1))
    
    return fig
        
def comp_violin_main(trainedmodel, controlmodel, runinfo):
    """Saves the violin plots comparing distribution of test scores for trained and control models 

    Arguments
    ---------
    trainedmodel : dict, information about trained model
    controlmodel : dict, information about control
    runinfo : RunInfo (extends dict)
    
    Returns
    -------
    
    """    
    
    trainedmodevals = get_modevals(trainedmodel, runinfo)
    controlmodevals = get_modevals(controlmodel, runinfo)
    
    ff = runinfo.analysisfolder(trainedmodel)
    
    fig = plot_compvp(trainedmodevals, controlmodevals, trainedmodel)
    
    os.makedirs('%s/comp_violin' %ff, exist_ok = True)
    fig.savefig('%s/comp_violin/comp_violin_v2_notypo_legcols_splitviolin.pdf' %(ff))
    
    print('figure saved')
    
    '''
    trainedmodevals_ee = get_modevals_ee(trainedmodel, runinfo)
    controlmodevals_ee = get_modevals_ee(controlmodel, runinfo)
    
    fig = plot_compvp_ee(trainedmodevals_ee, controlmodevals_ee, trainedmodel)
    
    
    os.makedirs('%s/comp_violin' %ff, exist_ok = True)
    fig.savefig('%s/comp_violin/comp_violin_v2_ee_notypo_legcols_splitviolin.pdf' %(ff))
    '''
    
    print('figure saved')
    
    plt.close('all')