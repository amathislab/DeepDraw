#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:57:21 2019

@auhor: kai

===CONTROLS LOG===
v4: with line through mean
v3: individual planes

===ROWWISE LOG===
v3: implemented subsampling and alpha for tuning curves,
    performed interpolation for contours in cartesian space (dir + vel tuning curves)
v2: separate training and test set
"""

#from pyglmnet import GLM
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from dir_tuning_alllayers_mp import *
from rowwise_neuron_curves_controls import *
import os

# %% FIND PREFERRED DIRECTIONS

def polarhist(dirs, ilayer, nbins=18, maxact = 14):
    if dirs != []:
        hist, binedges = np.histogram(dirs, bins=nbins, range=(-np.pi, np.pi))
        centers = [(binedges[i] + binedges[i + 1])/2 for i in range(nbins)]
        colors = [plt.cm.viridis(r/maxact) for r in hist]
    else:
        hist = []
        centers = []
        colors = []
    
    width = 2*np.pi/nbins
    fig = plt.figure( dpi=200)
    ax = plt.subplot(111, projection='polar')
    bars = ax.bar(centers, hist, width=width, color=colors)
    
    #cbar = fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=maxact), cmap=plt.cm.viridis), ax=ax)

    #cbar.set_ticks([(r/maxact) for r in range(maxact)])
    #cbar.set_yticklabels([r for r in range(maxact)])
    plt.title('Preferred Direction for Individual Neuron TCs L%d' %(ilayer), pad = 15)
    #plt.savefig('figs/%sl%d/l%d_prefdir_0%d_samecb.png' %(expf, ilayer, ilayer, int(r2threshold*10)))
    
    cbfig, cbax = plt.subplots(figsize=(1, 6))
    #cbfig.subplots_adjust(bottom=0.5)
    
    cmap = matplotlib.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=0, vmax=maxact)
    
    cb1 = matplotlib.colorbar.ColorbarBase(cbax, cmap=cmap,
                                    norm=norm,
                                    )
                                    #orientation='horizontal')
    #cbfig.savefig('figs/%sl%d/prefdir_0%d_colorbar_shared.png' %(expf, ilayer, int(r2threshold*10)))
        
    return fig, cbfig

def polarhistmean(dirs, ilayer, nbins=18, maxact = 14):
    if dirs != []:
        hist, binedges = np.histogram(dirs, bins=nbins, range=(-np.pi, np.pi))
        centers = [(binedges[i] + binedges[i + 1])/2 for i in range(nbins)]
        colors = [plt.cm.viridis(r/maxact) for r in hist]
    else:
        hist = []
        centers = []
        colors = []
    
    width = 2*np.pi/nbins
    fig = plt.figure( dpi=200)
    ax = plt.subplot(111, projection='polar')
    bars = ax.bar(centers, hist, width=width, color=colors)
    
    #cbar = fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=maxact), cmap=plt.cm.viridis), ax=ax)

    #cbar.set_ticks([(r/maxact) for r in range(maxact)])
    #cbar.set_yticklabels([r for r in range(maxact)])
    plt.title('Preferred Direction for Individual Neuron TCs L%d' %(ilayer), pad = 15)
    #plt.savefig('figs/%sl%d/l%d_prefdir_0%d_samecb.png' %(expf, ilayer, ilayer, int(r2threshold*10)))
    
    cbfig, cbax = plt.subplots(figsize=(1, 6))
    #cbfig.subplots_adjust(bottom=0.5)
    
    cmap = matplotlib.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=0, vmax=maxact)
    
    cb1 = matplotlib.colorbar.ColorbarBase(cbax, cmap=cmap,
                                    norm=norm,
                                    )
                                    #orientation='horizontal')
    #cbfig.savefig('figs/%sl%d/prefdir_0%d_colorbar_shared.png' %(expf, ilayer, int(r2threshold*10)))
    
    meanheight = np.nanmean(hist)
    theta =  np.arange(0, 2*np.pi, 0.01)
    r = [meanheight]*len(theta)
    
    ax.plot(theta, r, color='red', linewidth=5)    
    
    #plt.axis('off')
        
    return fig, cbfig

def main(model, runinfo, r2threshold = 0.2):
    
    modelname = model['name']    
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    base = model['base']
    
    fset = 'vel'
    mmod = 'std'
    
    resultsfolder = runinfo.resultsfolder(model)
    prefdirfolder = runinfo.analysisfolder(model, 'prefdir')
    expf = '%s%s/' %(resultsfolder, 'vel')
    
    # READ IN ALL EVALS
    alltestevals = []
    alltrainevals = []
        
    for ilayer in np.arange(0,nlayers):
        testevals = np.load('%sl%d_%s_mets_%s_%s_test.npy' %(expf, ilayer, fset, mmod, runinfo.planestring()))        
        alltestevals.append(testevals)
        
        trainevals = np.load('%sl%d_%s_mets_%s_%s_train.npy' %(expf, ilayer, fset, mmod, runinfo.planestring()))        
        alltrainevals.append(trainevals)
    
    #COMPUTE HISTOGRAMS 
    
    os.makedirs(prefdirfolder,exist_ok=True )
    
    for ilayer, testevals in enumerate(alltestevals):
        print("Layer %d" % (ilayer + 1))
        r2 = testevals[...,1,1]
        print(r2[r2 > r2threshold].size)
        sigfit = testevals[r2 > r2threshold]
        dirtuning = sigfit[...,1,3:5].reshape((-1,2))
        
        if len(dirtuning) > 0:
            prefdir = np.apply_along_axis(angle_xaxis, 1, dirtuning)
        else:
            prefdir = []
        prefdirfig, cbfig = polarhist(prefdir, ilayer, 18, model['max_act'])
        plt.tight_layout()
        prefdirfig.savefig(os.path.join(prefdirfolder, 'l%d_prefdir_0%d_18.pdf' %(ilayer, int(r2threshold*10))))
        cbfig.savefig(os.path.join(prefdirfolder, 'shared_colorbar.pdf'))
        plt.close('all')
        
        prefdirmeanfig, cbmeanfig = polarhistmean(prefdir, ilayer, 18, model['max_act'])
        plt.tight_layout()
        prefdirmeanfig.savefig(os.path.join(prefdirfolder, 'l%d_prefdir_meanheight_0%d_18.pdf' %(ilayer, int(r2threshold*10))))
        cbmeanfig.savefig(os.path.join(prefdirfolder, 'shared_colorbar_meanheight.pdf'))
