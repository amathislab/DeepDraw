#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:23:11 2019

@author: kai

v4: introduce decision rule on saved deviations
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from rowwise_neuron_curves_controls import *
import os

uniquezs = list(np.array([-45., -42., -39., -36., -33., -30., -27., -24., -21., -18., -15.,
                     -12.,  -9.,  -6.,  -3.,   0.,   3.,   6.,   9.,  12.,  15.,  18.,
                     21.,  24.,  27.,  30.]).astype(int))
uniquexs = list(np.array([ 6.,  9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42.,
                     45., 48., 51., 54.]).astype(int))
uniqueheights = [uniquezs, uniquexs]
orientations = ['hor', 'vert']

nzs = len(uniquezs)
nxs = len(uniquexs)
zpos0 = uniquezs.index(0)
xpos0 = int(len(uniquexs)/2)
x0 = uniquexs[xpos0]
z0 = 0

def format_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

# %% PREFERRED DIRECTION GENERALIZATION

def pdgenplot(dists, pds, model, desc):
    
    cm=matplotlib.cm.get_cmap('coolwarm')
    
    nlayers = model['nlayers'] + 1
    
    fig = plt.figure(figsize=(14,6), dpi=200)
    
    for ilayer in range(nlayers):
        plt.plot(dists, pds[ilayer], color=cm(ilayer/(nlayers-1)), marker='D')
    
    plt.ylim((-1.0,1.0))
    plt.title('preferred direction correlations \n %s' %desc)
    plt.xlabel('distance')
    plt.ylabel('correlation')
    plt.legend(['Spindles'] + ['Layer %d' %i for i in np.arange(1,nlayers+1)])
    plt.tight_layout()
    
    return fig

def prefdirgen(model, runinfo, r2threshold = 0.2):
    
    modelname = model['name']    
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    base = model['base']
    
    fset = 'vel'
    mmod = 'std'
    
    ##SAVE PDs & R2s
    pds = []
    for ilayer in range(nlayers):
        pds.append([])
        for ior, orientation in enumerate(orientations):
            pds[ilayer].append([])
            
            runinfo['orientation'] = orientation
            runinfo['height'] = uniqueheights[ior][0]
            
            resultsfolder = runinfo.resultsfolder(model)
            expf = '%s%s/' %(resultsfolder, 'vel')
            
            testevals = np.load('%sl%d_%s_mets_%s_%s_test.npy' %(expf, ilayer, fset, mmod, runinfo.planestring()))        
            dirtuning = testevals[...,1,3:5].reshape((-1,2))            
            prefdirs = np.apply_along_axis(angle_xaxis, 1, dirtuning)
            
            """Create numpy for pref dirs with structure:
                Ax 1: Height Index
                Ax 2: Neurons
            """
            pds[ilayer][ior] = np.zeros((len(uniqueheights[ior]), len(prefdirs)))
            
            for iht, ht in enumerate(uniqueheights[ior]):
                runinfo['height'] = ht

                testevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(runinfo.resultsfolder(model, 'vel'), ilayer, fset, mmod, runinfo.planestring()))
                dirtuning = testevals[...,1,3:5].reshape((-1,2))            
                prefdirs = np.apply_along_axis(angle_xaxis, 1, dirtuning)
                pds[ilayer][ior][iht] = prefdirs
            
    prefdirgenfolder = runinfo.generalizationfolder(model, 'prefdirgen')    
    os.makedirs(prefdirgenfolder,exist_ok=True )
            
    ## CALCULATE LAYER-WISE CORRELATIONS
    nzs = len(uniquezs)
    nxs = len(uniquexs)
    zpos0 = uniquezs.index(0)
    xpos0 = int(len(uniquexs)/2)
    x0 = uniquexs[xpos0]
    z0 = 0
    
    # HOR VS HOR
    r2s = pds
    horvshor = np.zeros((nlayers, nzs))
    for ilayer in range(nlayers):
        for iz in range(nzs):
            r20 = r2s[ilayer][0][zpos0]
            r2c = r2s[ilayer][0][iz]
            mask = np.logical_and(np.invert(np.isnan(r20)), np.invert(np.isnan(r2c)))
            horvshor[ilayer, iz] = np.corrcoef(r20[mask], r2c[mask])[0,1]

    fig = pdgenplot(uniquezs, horvshor, model, 'horizontal planes and plane at z=0')
    fig.savefig(os.path.join(prefdirgenfolder, 'prefdircomp_horvshor.png'))
    plt.close('all')
    
    # VERT VS VERT
    vertvsvert = np.zeros((nlayers, nxs))
    for ilayer in range(nlayers):
        for ix in range(nxs):
            r20 = r2s[ilayer][1][xpos0]
            r2c = r2s[ilayer][1][ix]
            mask = np.logical_and(np.invert(np.isnan(r20)), np.invert(np.isnan(r2c)))            
            vertvsvert[ilayer, ix] = np.corrcoef(r20[mask], r2c[mask])[0,1]
    
    fig = pdgenplot([x - x0 for x in uniquexs], vertvsvert, model, 'vertical planes and plane at x=%d' %x0)
    fig.savefig(os.path.join(prefdirgenfolder, 'prefdircomp_vertvsvert.png'))
    plt.close('all')
    
    #  HOR VS VERT (compare one horizontal to all verticals)
    horvsvert = np.zeros((nlayers, nxs))
    for ilayer in range(nlayers):
        for ix in range(nxs):
            r20 = r2s[ilayer][0][zpos0]
            r2c = r2s[ilayer][1][ix]
            mask = np.logical_and(np.invert(np.isnan(r20)), np.invert(np.isnan(r2c)))            
            horvsvert[ilayer, ix] = np.corrcoef(r20[mask], r2c[mask])[0,1]
    
    fig = pdgenplot(uniquexs, horvsvert, model, 'vertical planes and horizontal plane at z=0')
    fig.savefig(os.path.join(prefdirgenfolder, 'prefdircomp_horvsvert.png'))
    plt.close('all')
    
    # VERT VS HOR (one vertical vs. all horizontals)
    vertvshor = np.zeros((nlayers, nzs))
    for ilayer in range(nlayers):
        for iz in range(nzs):
            #print(pds[ilayer][1][xpos0].shape, pds[ilayer][0][iz].shape)
            r20 = r2s[ilayer][1][xpos0]
            r2c = r2s[ilayer][0][iz]
            mask = np.logical_and(np.invert(np.isnan(r20)), np.invert(np.isnan(r2c)))   
            vertvshor[ilayer, iz] = np.corrcoef(r20[mask], r2c[mask])[0,1]
    
    fig = pdgenplot(uniquezs, vertvshor, model, 'horizontal planes and vertical plane at x=%d' %x0 )
    fig.savefig(os.path.join(prefdirgenfolder, 'prefdircomp_vertvshor'))
    plt.close('all')
    
    
# %% R2 SCORE GENERALIZATION PLOTS
    
def r2genplot(dists, pds, model, desc):
    
    cm=matplotlib.cm.get_cmap('coolwarm')
    
    nlayers = model['nlayers'] + 1
    
    fig = plt.figure(figsize=(14,6), dpi=200)
    
    for ilayer in range(nlayers):
        plt.plot(dists, pds[ilayer], color=cm(ilayer/(nlayers-1)), marker='D')
    
    plt.title('r2 score correlations \n %s' %desc)
    plt.ylim((-1.0,1.0))
    plt.xlabel('distance')
    plt.ylabel('correlation')
    plt.legend(['Spindles'] + ['Layer %d' %i for i in np.arange(1,nlayers+1)])
    plt.tight_layout()
    
    return fig

def r2gen(model, runinfo, r2threshold = 0.2):
    
    modelname = model['name']    
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    base = model['base']
    
    fset = 'vel'
    mmod = 'std'
    
    ##SAVE R2s
    r2s = []
    for ilayer in range(nlayers):
        r2s.append([])
        for ior, orientation in enumerate(orientations):
            r2s[ilayer].append([])
            
            runinfo['orientation'] = orientation
            runinfo['height'] = uniqueheights[ior][0]
            
            resultsfolder = runinfo.resultsfolder(model)
            expf = '%s%s/' %(resultsfolder, 'vel')
            
            testevals = np.load('%sl%d_%s_mets_%s_%s_test.npy' %(expf, ilayer, fset, mmod, runinfo.planestring()))        

            r2 = testevals[...,1,1].reshape((-1))            
            
            """Create numpy for pref dirs with structure:
                Ax 1: Height Index
                Ax 2: Neurons
            """
            r2s[ilayer][ior] = np.zeros((len(uniqueheights[ior]), len(r2)))
            
            for iht, ht in enumerate(uniqueheights[ior]):
                runinfo['height'] = ht

                testevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(runinfo.resultsfolder(model, 'vel'), ilayer, fset, mmod, runinfo.planestring()))
                r2 = testevals[...,1,1].reshape((-1))            
                r2s[ilayer][ior][iht] = r2
            
    r2genfolder = runinfo.generalizationfolder(model, 'r2gen')    
    os.makedirs(r2genfolder,exist_ok=True )
            
    ## CALCULATE LAYER-WISE CORRELATIONS
    nzs = len(uniquezs)
    nxs = len(uniquexs)
    zpos0 = uniquezs.index(0)
    xpos0 = int(len(uniquexs)/2)
    x0 = uniquexs[xpos0]
    z0 = 0
    
    # HOR VS HOR
    horvshor = np.zeros((nlayers, nzs))
    for ilayer in range(nlayers):
        for iz in range(nzs):
            r20 = r2s[ilayer][0][zpos0]
            r2c = r2s[ilayer][0][iz]
            mask = np.logical_and(np.invert(np.isnan(r20)), np.invert(np.isnan(r2c)))
            horvshor[ilayer, iz] = np.corrcoef(r20[mask], r2c[mask])[0,1]

    fig = r2genplot(uniquezs, horvshor, model, 'horizontal planes and plane at z=0')
    fig.savefig(os.path.join(r2genfolder, 'r2comp_horvshor.png'))
    plt.close('all')
    
    # VERT VS VERT
    vertvsvert = np.zeros((nlayers, nxs))
    for ilayer in range(nlayers):
        for ix in range(nxs):
            r20 = r2s[ilayer][1][xpos0]
            r2c = r2s[ilayer][1][ix]
            mask = np.logical_and(np.invert(np.isnan(r20)), np.invert(np.isnan(r2c)))            
            vertvsvert[ilayer, ix] = np.corrcoef(r20[mask], r2c[mask])[0,1]
    
    fig = r2genplot([x - x0 for x in uniquexs], vertvsvert, model, 'vertical planes and plane at x=%d' %x0)
    fig.savefig(os.path.join(r2genfolder, 'r2comp_vertvsvert.png'))
    plt.close('all')
    
    #  HOR VS VERT (compare one horizontal to all verticals)
    horvsvert = np.zeros((nlayers, nxs))
    for ilayer in range(nlayers):
        for ix in range(nxs):
            r20 = r2s[ilayer][0][zpos0]
            r2c = r2s[ilayer][1][ix]
            mask = np.logical_and(np.invert(np.isnan(r20)), np.invert(np.isnan(r2c)))            
            horvsvert[ilayer, ix] = np.corrcoef(r20[mask], r2c[mask])[0,1]
    
    fig = r2genplot(uniquexs, horvsvert, model, 'vertical planes and horizontal plane at z=0')
    fig.savefig(os.path.join(r2genfolder, 'r2comp_horvsvert.png'))
    plt.close('all')
    
    # VERT VS HOR (one vertical vs. all horizontals)
    vertvshor = np.zeros((nlayers, nzs))
    for ilayer in range(nlayers):
        for iz in range(nzs):
            r20 = r2s[ilayer][1][xpos0]
            r2c = r2s[ilayer][0][iz]
            mask = np.logical_and(np.invert(np.isnan(r20)), np.invert(np.isnan(r2c)))   
            vertvshor[ilayer, iz] = np.corrcoef(r20[mask], r2c[mask])[0,1]
    
    fig = r2genplot(uniquezs, vertvshor, model, 'horizontal planes and vertical plane at x=%d' %x0 )
    fig.savefig(os.path.join(r2genfolder, 'r2comp_vertvshor'))
    plt.close('all')
    
# %% INDIVIDUAL NEURON INVARIANCE
    
def plot_ind_neuron_invar(pds, hts, layer, orientation):
    
    nns = len(pds[0])
    fig = plt.figure(figsize = (10, nns/5), dpi=50)
    
    
    for neuron in range(len(pds[0])):
        plt.plot(hts, [pds[i][neuron]/(2*np.pi) + neuron for i in range(len(hts))], alpha = 0.5)
    
    plt.yticks(range(nns))
    plt.title('Invariance at an Individual Neuron Level for L%d %s' %(layer, orientation))
    plt.xlabel('Plane Height')
    plt.ylabel('Preferred Directions')
    
    return fig

def normalized_angle_calculator(angle, angle_pos0):
    norm = angle - angle_pos0
    if(norm > np.pi):
        norm = norm - 2*np.pi
    if(norm < - np.pi):
        norm = norm + 2*np.pi
    return(norm)

def plot_ind_neuron_invar_collapsed_beautified(pds, hts, layer, orientation):
    
    nns = len(pds[0])
    fig = plt.figure(figsize=(8,6),dpi=300)
    
    if orientation == 'hor':
        pos0 = zpos0
        hstr = 'z = %d' %z0
    else:
        pos0 = xpos0
        hstr = 'x = %d' %x0
        
    nns = len(pds[0])    
    neuronpds = []
    
    for neuron in range(len(pds[0])):
        plt.plot(hts, [normalized_angle_calculator(pds[i][neuron], pds[pos0][neuron]) for i in range(len(hts))], alpha=0.6, color='grey')
    
        neuronpds.append([normalized_angle_calculator(pds[i][neuron], pds[pos0][neuron]) for i in range(len(hts))])

    neuronpds = np.vstack(neuronpds).astype('float64')
    
    plt.plot(hts, np.nanmean(neuronpds,axis=0), linewidth=3, color='red')
    
    plt.ylim((-np.pi, np.pi))

    if orientation == 'hor':
        plt.xlim((-45, 35))
    else:
        plt.xlim((3, 57))
        
    plt.xlabel('Plane Height')
    plt.ylabel('Deviation from Plane at %s' %hstr)
    
    format_axis(plt.gca())
    
    deviations= np.nanmean(np.abs(neuronpds),axis=0)
    
    #set all planes where only a single neuron is directionally tuned to np.nan
    greaterequal_2 = np.sum(~np.isnan(neuronpds),axis=0)
    print('number of neurons that are directionally tuned in each plane: ', greaterequal_2)
    greaterequal_2 = np.where(greaterequal_2 >= 3, True, False)
    
    deviations[~greaterequal_2] = np.nan
    
    return fig, deviations
    
def ind_neuron_invar(model, runinfo, r2threshold = 0.2):
    modelname = model['name']    
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    base = model['base']
    
    fset = 'vel'
    mmod = 'std'
    
    ##SAVE PDs & R2s
    pds = []
    
    for ilayer in range(nlayers):
        pds.append([])
        for ior, orientation in enumerate(orientations):
            pds[ilayer].append([])
            
            runinfo['orientation'] = orientation
            runinfo['height'] = uniqueheights[ior][0]
            
            resultsfolder = runinfo.resultsfolder(model)
            expf = '%s%s/' %(resultsfolder, 'vel')
            
            testevals = np.load('%sl%d_%s_mets_%s_%s_test.npy' %(expf, ilayer, fset, mmod, runinfo.planestring()))        
            dirtuning = testevals[...,1,3:5].reshape((-1,2))            
            prefdirs = np.apply_along_axis(angle_xaxis, 1, dirtuning)
            
            """Create numpy for pref dirs with structure:
                Ax 1: Height Index
                Ax 2: Neurons
            """
            pds[ilayer][ior] = np.zeros((len(uniqueheights[ior]), len(prefdirs)))
            
            for iht, ht in enumerate(uniqueheights[ior]):
                runinfo['height'] = ht

                testevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(runinfo.resultsfolder(model, 'vel'), ilayer, fset, mmod, runinfo.planestring()))
                dirtuning = testevals[...,1,3:5].reshape((-1,2))
                        
                prefdirs = np.apply_along_axis(angle_xaxis, 1, dirtuning)
                pds[ilayer][ior][iht] = prefdirs
    
    ffolder = runinfo.generalizationfolder(model, 'ind_neuron_invar')
    os.makedirs(ffolder, exist_ok=True)
    
    for ilayer in range(nlayers):
        print(ilayer)
        for ior, orientation in enumerate(orientations):
            fig = plot_ind_neuron_invar(pds[ilayer][ior], uniqueheights[ior], ilayer, orientation)
            fig.savefig(os.path.join(ffolder, 'ind_neuron_invar_l%d_%s.png' %(ilayer, orientation)))
            plt.close('all')
            
def ind_neuron_invar_collapsed(model, runinfo, r2threshold = 0.2):
    modelname = model['name']    
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    base = model['base']
    
    fset = 'vel'
    mmod = 'std'
    
    ##SAVE PDs & R2s
    pds = []
    
    for ilayer in range(nlayers):
        pds.append([])
        for ior, orientation in enumerate(orientations):
            pds[ilayer].append([])
            
            runinfo['orientation'] = orientation
            runinfo['height'] = uniqueheights[ior][0]
            
            resultsfolder = runinfo.resultsfolder(model)
            expf = '%s%s/' %(resultsfolder, 'vel')
            
            testevals = np.load('%sl%d_%s_mets_%s_%s_test.npy' %(expf, ilayer, fset, mmod, runinfo.planestring()))        
            dirtuning = testevals[...,1,3:5].reshape((-1,2))            
            prefdirs = np.apply_along_axis(angle_xaxis, 1, dirtuning)
            
            """Create numpy for pref dirs with structure:
                Ax 1: Height Index
                Ax 2: Neurons
            """
            pds[ilayer][ior] = np.zeros((len(uniqueheights[ior]), len(prefdirs)))
            
            for iht, ht in enumerate(uniqueheights[ior]):
                runinfo['height'] = ht

                testevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(runinfo.resultsfolder(model, 'vel'), ilayer, fset, mmod, runinfo.planestring()))
                dirtuning = testevals[...,1,3:5].reshape((-1,2))
    
                #apply r2threshold
                for neuron, score in enumerate(testevals[...,1,1].flatten()):
                    if score < r2threshold:
                        dirtuning[neuron] = np.nan
                
                prefdirs = np.apply_along_axis(angle_xaxis, 1, dirtuning)
                pds[ilayer][ior][iht] = prefdirs
    
    ffolder = runinfo.generalizationfolder(model, 'ind_neuron_invar_collapsed_beautified')
    os.makedirs(ffolder, exist_ok=True)
    
    for ior, orientation in enumerate(orientations):
        indinvars_index = range(nlayers)
        indinvars_columns = uniqueheights[ior]
        indinvars_df = pd.DataFrame(index = indinvars_index, columns = indinvars_columns)
        for ilayer in range(nlayers):
            print(ilayer)
            
            fig, deviations = plot_ind_neuron_invar_collapsed_beautified(pds[ilayer][ior], uniqueheights[ior], ilayer, orientation)
            fig.savefig(os.path.join(ffolder, 'ind_neuron_invar_l%d_%s_collapsed_0%d_v2.pdf' %(ilayer, orientation, int(r2threshold*10))))
            plt.close('all')
            print('indinvars_collapsed plots plotted')
            
            indinvars_df.loc[ilayer] = deviations
            
        indinvars_df.to_csv(os.path.join(ffolder, 'ind_neuron_invar_%s_deviations_0%d.csv' %(orientation, int(r2threshold*10))))
            
# %% GENERALIZATION MAIN

def main(model, runinfo, r2threshold = 0.2):
    print('creating prefdir correlation generalization plot for model %s...' %model['name'])
    if(not os.path.exists(runinfo.generalizationfolder(model, 'prefdirgen'))):
    #if(True):
        prefdirgen(model, runinfo, r2threshold)
    else:
        print('prefdir correlation generalization plot already exists')

    print('creating r2 score correlation generalization plot for model %s ...' %model['name'])
    if(not os.path.exists(runinfo.generalizationfolder(model, 'r2gen'))):
        r2gen(model, runinfo, r2threshold)
    else:
        print('r2 score generalization plot already exists')
        
    print('creating individual neuron generalization plot for model %s ...' %model['name'])
    if(not os.path.exists(runinfo.generalizationfolder(model, 'ind_neuron_invar'))):
    #if(True):
        ind_neuron_invar(model, runinfo)
        print('plots saved')
    else:
        print('individual neuron invariance plot already created')
                
    if(not os.path.exists(runinfo.generalizationfolder(model, 'ind_neuron_invar_collapsed_beautified'))):
    #if(True):
        ind_neuron_invar_collapsed(model, runinfo)
        print('plots saved')
    else:
        print('individual neuron collapsed invariance plot already created')
