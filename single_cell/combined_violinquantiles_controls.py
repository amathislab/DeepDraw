#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:23:52 2019

@author: kai

v7: ee comparison
v6: split violin plot
v3: individual planes
v2: restructured folders
"""

#from pyglmnet import GLM
import numpy as np
import matplotlib
matplotlib.use('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
#from dir_tuning_alllayers_mp import *
from rowwise_neuron_curves_controls import *
import pickle
import os
from scipy.interpolate import griddata

def format_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

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
expid = 100
fset = 'vel'
mmod = 'std'
tcoff = 32

#temporal spatial model

modnames = ['Dir', 'Vel', 'Dir x Vel', 'Acc', 'Labels']
nmods = len(modnames)

# %%

def get_modevals(model, runinfo):
    
    expf={
          'vel': runinfo.resultsfolder(model, 'vel'),
          'acc': runinfo.resultsfolder(model, 'acc'),
          'labels': runinfo.resultsfolder(model, 'labels')
    }
    
    '''
    #Spatial temporal model
    expf={
          'vel': '%s/%s/exp%d/' %(modelname, 'vel', 7),
          'acc': '%s/%s/exp%d/' %(modelname, 'acc', 8),
          'labels': '%s/%s/exp%d/' %(modelname, 'labels', 1)
    }
    '''
    
    # %% READ IN OF REGS AND THRESHOLD, SAVE TEXT FILE
    

    
    modevals = []
    
    for i in range(nmods):
        modevals.append([])
    
    #r2threshold = 0.2
    
	#f= open("guru99.txt","w+")
    #f.write()
    
    for ilayer in np.arange(0,model['nlayers'] + 1):
        
        dvevals = np.load(os.path.join(expf['vel'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'vel', mmod, runinfo.planestring())))
        accevals = np.load(os.path.join(expf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std',runinfo.planestring())))
        labevals = np.load(os.path.join(expf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
        
        modevals[0].append(dvevals[...,1,1]) #dir
        modevals[1].append(dvevals[...,2,1]) #vel
        modevals[2].append(dvevals[...,3,1]) #dir + vel
        modevals[3].append(accevals[...,2,1]) #acc
        modevals[4].append(labevals[:,0]) #labels
        #modevals[4].append(labevals[...,21,1]) #labels
        
        
    return modevals

def main(model, runinfo):
    
    modelname = model['name']
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    base = model['base']
    #modeltype = model['type']
    
    cmapnames = dict({'S': 'Blues_r',
                      'T': 'Oranges_r',
                      'ST': 'Greens_r'})
    
    cmap = matplotlib.cm.get_cmap(cmapnames[model['type']]) #spatial_temporal    
    
    ff = runinfo.analysisfolder(model)
    
    #resultsfolder = runinfo.resultsfolder(model)
    modevals = get_modevals(model, runinfo)
        
    # %% SHARED VIOLIN PLOT
           
    def combvipqs(modevals, runfinfo):
        
        #plot parameters
        #color = ['blue', 'red', 'green', 'orange']
            
        space = 0.8
        width = 0.6
        lspace = space*nmods + 1
        
        ct = 0.4
        cidx = [i*(1-ct)/(nmods-1) for i in range(nmods)] #Blues_r option
        #cidx = [i*0.60/(nmods-1) + 0.40 for i in range(nmods)] 
            
        #plot figure
        fig = plt.figure(figsize=(14,6), dpi=200)
        ax1 = fig.add_subplot(111)
        
        #plt.title('Distribution and 90% Quantiles of r2 Scores for Different Kin Model Types')
        
        plt.xticks(np.arange(lspace/2,nlayers*lspace + 1,lspace), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,nlayers+1)])
                       #horizontalalignment='left')
        #ax2 = ax1.twinx()
        
        ax1.set_ylabel('r2 or classification score')
        ax1.set_ylim(-0.1, 1.1)
        #ax2.set_ylim(0.45, 1.05)
        #ax2.set_ylabel('roc auc score')
        #[at]
        vps = []
        patches = []
        for i, mod in enumerate(modevals):
            """if(i < 4):
                ax = ax1
            else:
                ax = ax2"""
                
            vp = ax1.violinplot(mod,
                positions=[ilayer*lspace+space*i+1 for ilayer in range(nlayers)], 
                #c = 'C%d' %i,
                widths=width)
            
            #cidx = i*0.60/(nmods-1) + 0.40
            #cidx = i*0.40/(nmods-1) + 0.60
            #print(cidx)
            
            ''' CMAP values thus chosen:
            0.4
            0.55
            0.7
            0.85
            1.0
            '''
            
            for part in vp['bodies']:
                part.set_facecolor(cmap(cidx[i]))
                part.set_edgecolor(cmap(cidx[i]))
            vp['cbars'].set_color(cmap(cidx[i]))
            vp['cmins'].set_color(cmap(cidx[i]))
            vp['cmaxes'].set_color(cmap(cidx[i]))
            
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
                
            #cidx = i*0.60/(nmods-1) + 0.40
            
            ax1.plot([ilayer*lspace+space*i+1 for ilayer in range(nlayers)], 
                      q90s, color=cmap(cidx[i]), marker=marker)
        
        #plt.xlim(0, nlayers * lspace + 1)
    
        plt.legend(patches, modnames, loc='upper right')
        
        format_axis(plt.gca())
        plt.savefig('%s/joint/combviolinq90s_%s_test_r_fixedtypo.png' %(ff, runinfo.planestring()))
        plt.close('all')
        
    os.makedirs('%s/joint' %ff, exist_ok = True)
    
    combvipqs(modevals, runinfo)
    print('plot saved')

# %% COMPARISON VIOLIN PLOT 


def clip(vp, lr):
    
    for b in vp['bodies']:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        #print(b.get_paths()[0].vertices[:, 0])
    
        if lr == 'l':
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        elif lr == 'r':
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            
        #print(b.get_paths()[0].vertices[:, 0])
    
        
    '''
    for b in [vp['cbars'], vp['cmins'], vp['cmaxes']]:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        #print(b.get_paths()[0].vertices[:, 0])
    
        if lr == 'l':
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        elif lr == 'r':
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    '''
    
           
def plot_compvp(trainedmodevals, controlmodevals, trainedmodel):
        
    #plot parameters
    #color = ['blue', 'red', 'green', 'orange']
    
    nlayers = trainedmodel['nlayers'] + 1
    
    space = 0.8
    width = 0.6
    lspace = space*nmods + 1
    
    trainedcmap = matplotlib.cm.get_cmap(trainedmodel['cmap']) #spatial_temporal    
    controlcmap = matplotlib.cm.get_cmap('Greys_r') #spatial_temporal    
    cmaps = [trainedcmap, controlcmap]
    
    ct = 0.6
    cidx = [i*(1-ct)/(nmods-1) for i in range(nmods)] #Blues_r option
    #cidx = [i*0.60/(nmods-1) + 0.40 for i in range(nmods)] 
        
    #plot figure
    fig = plt.figure(figsize=(14,6), dpi=200)
    ax1 = fig.add_subplot(111)
    
    #plt.title('Distribution and 90% Quantiles of r2 Scores for Different Kin Model Types')
    
    plt.xticks(np.arange(lspace/2,nlayers*lspace + 1,lspace), 
               ['Sp.'] + ['L%d' %i for i in np.arange(1, nlayers+1)])
               #['Spindles'] + ['Layer %d' %i for i in np.arange(1,nlayers+1)])
                   #horizontalalignment='left')
    #ax2 = ax1.twinx()
    
    ax1.set_ylabel('r2 or classification score')
    ax1.set_ylim(-0.1, 1.1)
    #ax2.set_ylim(0.45, 1.05)
    #ax2.set_ylabel('roc auc score')
    #[at]
    vps = []
    patches = []
    
    ccolorindex = 3
    #for (modevals, cmap, alpha) in zip([trainedmodevals, controlmodevals], cmaps, [[1, 0.5], [0.7, 0.2]]):
    for (modevals, cmap, alpha, zorder, lr) in zip([controlmodevals, trainedmodevals], [controlcmap, trainedcmap], [[0.8, 0.5], [0.8, 0.7]], [2,1], ['r', 'l']):
    #for (modevals, cmap, alpha, zorder, lr) in zip([controlmodevals, trainedmodevals], [controlcmap, trainedcmap], [[1.0, 1.0], [1.0, 1.0]], [2,1], ['r', 'l']):
        for i, mod in enumerate(modevals):
            """if(i < 4):
                ax = ax1
            else:
                ax = ax2"""
                
           #mod = [lm.reshape((-1,)) for lm in mod]
            #print(i, mod)
                
            vp = ax1.violinplot(mod,
                positions=[ilayer*lspace+space*i+1 for ilayer in range(nlayers)], 
                showextrema = False,
                showmedians = False,
                showmeans = False,
                #c = 'C%d' %i,
                widths=width)
                #zorder=zorder)
            
            #cidx = i*0.60/(nmods-1) + 0.40
            #cidx = i*0.40/(nmods-1) + 0.60
            #print(cidx)
            
            ''' CMAP values thus chosen:
            0.4
            0.55
            0.7
            0.85
            1.0
            '''
            
            for part in vp['bodies']:
                if lr == 'l':
                    part.set_facecolor(cmap(cidx[i]))
                    part.set_edgecolor(cmap(cidx[i]))
                else:
                    part.set_facecolor(cmap(cidx[ccolorindex]))
                    part.set_edgecolor(cmap(cidx[ccolorindex]))
                part.set_alpha(alpha[1])
                part.set_zorder(zorder)
            
            '''
            vp['cbars'].set_color(cmap(cidx[i]))
            vp['cmins'].set_color(cmap(cidx[i]))
            vp['cmaxes'].set_color(cmap(cidx[i]))
            vp['cbars'].set_alpha(alpha[0])
            vp['cmins'].set_alpha(alpha[0])
            vp['cmaxes'].set_alpha(alpha[0])
            vp['cbars'].set_zorder(zorder)
            vp['cmins'].set_zorder(zorder)
            vp['cmaxes'].set_zorder(zorder)
            '''
            
            
            '''
            for b in vp['bodies']:
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                if lr == 'l':
                    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                elif lr == 'r':
                    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                else:
                    assert 1 == 0, 'error'
                #b.set_color('r')
            '''
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
                
            #cidx = i*0.60/(nmods-1) + 0.40
            
            ax1.plot([ilayer*lspace+space*i+1 for ilayer in range(nlayers)], 
                      q90s, color=cmap(cidx[i]), marker=marker, alpha=alpha[0])
    
    #plt.xlim(0, nlayers * lspace + 1)
    format_axis(plt.gca())
    leg = plt.legend(patches[5:], modnames, loc='upper right')
    ax1.add_artist(leg)
    plt.legend([patches[5], patches[ccolorindex]], ['Trained', 'Control'], loc='upper right', bbox_to_anchor=(0.87, 1))
    
    return fig

def get_modevals_ee(model, runinfo):
    
    expf={
          'ee': runinfo.resultsfolder(model, 'ee'),
          #'acc': runinfo.resultsfolder(model, 'acc'),
          #'labels': runinfo.resultsfolder(model, 'labels')
    }
    
    '''
    #Spatial temporal model
    expf={
          'vel': '%s/%s/exp%d/' %(modelname, 'vel', 7),
          'acc': '%s/%s/exp%d/' %(modelname, 'acc', 8),
          'labels': '%s/%s/exp%d/' %(modelname, 'labels', 1)
    }
    '''
    
    # %% READ IN OF REGS AND THRESHOLD, SAVE TEXT FILE
    

    
    modevals = []
    
    nmods_ee = 2
    
    for i in range(nmods_ee):
        modevals.append([])
    
    #r2threshold = 0.2
    
	#f= open("guru99.txt","w+")
    #f.write()
    
    for ilayer in np.arange(0,model['nlayers'] + 1):
        
        eeevals = np.load(os.path.join(expf['ee'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'ee', mmod, runinfo.planestring())))
        #accevals = np.load(os.path.join(expf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std',runinfo.planestring())))
        #labevals = np.load(os.path.join(expf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
        
        modevals[0].append(eeevals[...,0,1]) #dir
        #modevals[1].append(dvevals[...,2,1]) #vel
        modevals[1].append(eeevals[...,3,1]) #dir + vel
        #modevals[3].append(accevals[...,2,1]) #acc
        #modevals[4].append(labevals[:,0]) #labels
        #modevals[4].append(labevals[...,21,1]) #labels
        
        
    return modevals


def plot_compvp_ee(trainedmodevals, controlmodevals, trainedmodel):
        
    #plot parameters
    #color = ['blue', 'red', 'green', 'orange']
    
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
    #cidx = [i*0.60/(nmods-1) + 0.40 for i in range(nmods)] 
        
    #plot figure
    fig = plt.figure(figsize=(14,6), dpi=200)
    ax1 = fig.add_subplot(111)
    
    #plt.title('Distribution and 90% Quantiles of r2 Scores for Different Kin Model Types')
    
    plt.xticks(np.arange(lspace/2,nlayers*lspace + 1,lspace), 
               ['Sp.'] + ['L%d' %i for i in np.arange(1, nlayers+1)])
               #['Spindles'] + ['Layer %d' %i for i in np.arange(1,nlayers+1)])
                   #horizontalalignment='left')
    #ax2 = ax1.twinx()
    
    ax1.set_ylabel('r2 or classification score')
    ax1.set_ylim(-0.1, 1.1)
    #ax2.set_ylim(0.45, 1.05)
    #ax2.set_ylabel('roc auc score')
    #[at]
    vps = []
    patches = []
    
    ccolorindex = 1
    #for (modevals, cmap, alpha) in zip([trainedmodevals, controlmodevals], cmaps, [[1, 0.5], [0.7, 0.2]]):
    for (modevals, cmap, alpha, zorder, lr) in zip([controlmodevals, trainedmodevals], [controlcmap, trainedcmap], [[0.8, 0.5], [0.8, 0.7]], [2,1], ['r', 'l']):
    #for (modevals, cmap, alpha, zorder, lr) in zip([controlmodevals, trainedmodevals], [controlcmap, trainedcmap], [[1.0, 1.0], [1.0, 1.0]], [2,1], ['r', 'l']):
        for i, mod in enumerate(modevals):
            """if(i < 4):
                ax = ax1
            else:
                ax = ax2"""
                
            #print(i, mod)
            #mod = [np.unique(lm.reshape((-1,)).astype('float64')) for lm in mod]
            #print(i, mod)
                
            vp = ax1.violinplot(mod,
                positions=[ilayer*lspace+space*i+1 for ilayer in range(nlayers)], 
                showextrema = False,
                showmedians = False,
                showmeans = False,
                #c = 'C%d' %i,
                widths=width)
                #zorder=zorder)
            
            #cidx = i*0.60/(nmods-1) + 0.40
            #cidx = i*0.40/(nmods-1) + 0.60
            #print(cidx)
            
            ''' CMAP values thus chosen:
            0.4
            0.55
            0.7
            0.85
            1.0
            '''
            
            for part in vp['bodies']:
                if lr == 'l':
                    part.set_facecolor(cmap(cidx[i]))
                    part.set_edgecolor(cmap(cidx[i]))
                else:
                    part.set_facecolor(cmap(cidx[ccolorindex]))
                    part.set_edgecolor(cmap(cidx[ccolorindex]))
                part.set_alpha(alpha[1])
                part.set_zorder(zorder)
            
            '''
            vp['cbars'].set_color(cmap(cidx[i]))
            vp['cmins'].set_color(cmap(cidx[i]))
            vp['cmaxes'].set_color(cmap(cidx[i]))
            vp['cbars'].set_alpha(alpha[0])
            vp['cmins'].set_alpha(alpha[0])
            vp['cmaxes'].set_alpha(alpha[0])
            vp['cbars'].set_zorder(zorder)
            vp['cmins'].set_zorder(zorder)
            vp['cmaxes'].set_zorder(zorder)
            '''
            
            
            '''
            for b in vp['bodies']:
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                if lr == 'l':
                    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                elif lr == 'r':
                    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                else:
                    assert 1 == 0, 'error'
                #b.set_color('r')
            '''
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
                
            #cidx = i*0.60/(nmods-1) + 0.40
            
            ax1.plot([ilayer*lspace+space*i+1 for ilayer in range(nlayers)], 
                      q90s, color=cmap(cidx[i]), marker=marker, alpha=alpha[0])
    
    #plt.xlim(0, nlayers * lspace + 1)
    format_axis(plt.gca())
    leg = plt.legend(patches[2:], modnames_ee, loc='upper right')
    ax1.add_artist(leg)
    plt.legend([patches[2], patches[ccolorindex]], ['Trained', 'Control'], loc='upper right', bbox_to_anchor=(0.87, 1))
    
    return fig
        
def comp_violin_main(trainedmodel, controlmodel, runinfo):
    
    trainedmodevals = get_modevals(trainedmodel, runinfo)
    controlmodevals = get_modevals(controlmodel, runinfo)
    
    ff = runinfo.analysisfolder(trainedmodel)
    
    fig = plot_compvp(trainedmodevals, controlmodevals, trainedmodel)
    
    os.makedirs('%s/comp_violin' %ff, exist_ok = True)
    fig.savefig('%s/comp_violin/comp_violin_v2_notypo_legcols_splitviolin.pdf' %(ff))
    
    print('figure saved')
    
    trainedmodevals_ee = get_modevals_ee(trainedmodel, runinfo)
    controlmodevals_ee = get_modevals_ee(controlmodel, runinfo)
    
    #print(trainedmodevals_ee)
    
    #ff = runinfo.analysisfolder(trainedmodel)
    
    fig = plot_compvp_ee(trainedmodevals_ee, controlmodevals_ee, trainedmodel)
    
    os.makedirs('%s/comp_violin' %ff, exist_ok = True)
    fig.savefig('%s/comp_violin/comp_violin_v2_ee_notypo_legcols_splitviolin.pdf' %(ff))
    
    print('figure saved')
    
    plt.close('all')
