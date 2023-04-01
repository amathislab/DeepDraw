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
    ''' format axis to unified format
    
    Arguments
    ---------
    ax : matplotlib axis object
    '''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)
    
# %% INDIVIDUAL NEURON INVARIANCE

def normalized_angle_calculator(angle, angle_pos0):
    """Returns the difference between two angles (between -pi and pi)
    
    Arguments
    ---------
    angle : float, angle to be normalized
    angle_pos0 : float, angle that we are normalizing by
    
    Returns
    -------
    norm : float, normalized angle
    """
    
    
    norm = angle - angle_pos0
    if(norm > np.pi):
        norm = norm - 2*np.pi
    if(norm < - np.pi):
        norm = norm + 2*np.pi
    return(norm)

def plot_ind_neuron_invar_collapsed_beautified(pds, hts, orientation):
    """ Plot deviation plots showing the deviation in preferred angle of a neuron from that at the central plane
    
    Arguments
    ---------
    pds : list of lists, outer list : plane, inner list: neuron, content : preferred direction
    hts : list of ints, possible height of planes
    orientation : str, 'hor' or 'vert'
    
    Returns
    -------
    fig : plt.figure, collapsed ind neuron deviation plot
    deviations : np.array [nr_planes,]
    
    """
    
    fig = plt.figure(figsize=(8,6),dpi=300)
    
    if orientation == 'hor':
        pos0 = zpos0
        hstr = 'z = %d' %z0
    else:
        pos0 = xpos0
        hstr = 'x = %d' %x0
        
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
    
def ind_neuron_invar_collapsed(model, runinfo, r2threshold = 0.2):
    '''Calculate and plot deviation of single neurons from invariance
    
    Arguments
    ---------
    model : dict
    runinfo : RunInfo (extension of dict)
    r2threshold : float, threshold above which test scores mean a neuron is directionally tuned
       
    '''  
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    
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
            pds[ilayer][ior][:] = np.NaN
            
            for iht, ht in enumerate(uniqueheights[ior]):
                runinfo['height'] = ht

                testevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(runinfo.resultsfolder(model, 'vel'), ilayer, fset, mmod, runinfo.planestring()))
                dirtuning = testevals[...,1,3:5].reshape((-1,2))
    
                #apply r2threshold
                for neuron, score in enumerate(testevals[...,1,1].flatten()):
                    if score < r2threshold:
                        dirtuning[neuron] = np.nan
                
                if(len(dirtuning) > 0):
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
            fig.savefig(os.path.join(ffolder, 'ind_neuron_invar_l%d_%s_collapsed_0%d_v2.png' %(ilayer, orientation, int(r2threshold*10))))
            fig.savefig(os.path.join(ffolder, 'ind_neuron_invar_l%d_%s_collapsed_0%d_v2.svg' %(ilayer, orientation, int(r2threshold*10))))
            plt.close('all')
            print('indinvars_collapsed plots plotted')
            
            indinvars_df.loc[ilayer] = deviations
            
        indinvars_df.to_csv(os.path.join(ffolder, 'ind_neuron_invar_%s_deviations_0%d.csv' %(orientation, int(r2threshold*10))))
            
# %% GENERALIZATION MAIN

def main(model, runinfo, r2threshold = 0.2):        
    print('creating individual neuron generalization plot for model %s ...' %model['name'])
                
    #if(runinfo.default_run):
    #if(not os.path.exists(runinfo.generalizationfolder(model, 'ind_neuron_invar_collapsed_beautified'))):
    if(True):
        ind_neuron_invar_collapsed(model, runinfo, r2threshold)
        print('plots saved')
    else:
        print('individual neuron collapsed invariance plot already created')
