#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:20:57 2019
Majorly revamped on Sun Jan 31 2021

@author: kai
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from skimage.transform import resize

from rowwise_neuron_curves_controls import read_layer_reps, X_data

# %% SETUP
#PARS
modelname = 'spatial_temporal_4_8-16-16-32_64-64-64-64_5272'
datafolder = '../data/%s/' %modelname

char_labels = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']
nchars = len(char_labels)

topk = 5

# %% FUNCTIONS
def get_max_act(actmap):
    ''' return the maximum activation across an activation map '''
    return np.max(actmap.flatten())

def topk_actmaps(actmaps, labels):
    ''' return the indices corresponding to top k activations across an activation map '''
    topkam = np.zeros(tuple([nchars, topk]) + actmaps[0].shape)
    idxs = np.zeros((nchars, topk))
    for idx, am in enumerate(actmaps):
        char = labels[idx]
        maxact = get_max_act(am)
        #print(idx, maxact, )
        if maxact > get_max_act(topkam[char, topk - 1]):
            topkam[char, topk - 1] = am
            maxacts = np.array([get_max_act(tam) for tam in topkam[char]])
            order = np.argsort(- maxacts)
            #print(order)
            topkam[char] = topkam[char, order]
            
            idxs[char, topk - 1] = idx
            idxs[char] = idxs[char, order]
    return topkam, idxs

        
# %% PLOT & ANALYZE RESULTS
def above_threshold(am, threshold = 0.3):
    """Returns boolean map where activation is above 0.3 * maxact """
    maxact = get_max_act(am)
    bat = np.where(am > maxact * threshold, True, False)
    return bat

def get_t(fmapt, fmapnt, nt = 320, t_stride = 2):
    """Converts time dimension index from feature map to original time point"""
    centers = np.arange(nt)
    for i in range(int(np.log2(len(centers)/ fmapnt))):
        centers = [centers[i*t_stride] for i in range(len(centers)//2)]
    
    assert len(centers) == fmapnt, "length of centers and fmapnt not equal!"
    
    return centers[fmapt]

def get_s(fmaps, fmapns, ns = 25, s_stride = 2):
    """Converts spatial dimension index from feature map to original time point"""
    centers = np.arange(ns)
    while len(centers) > fmapns:
        centers = [centers[i*s_stride] for i in range((len(centers)+1)//2)]
    
    assert len(centers) == fmapns, "length of centers and fmapnt not equal!"
    
    return centers[fmaps]

def get_ext(ext, fmapn, n = 320, stride = 2):
    cext = 0
    centers = []
    centers.append([i for i in range(n)])
    while(len(centers[-1]) > fmapn):
        cs = centers[-1]
        centers.append([cs[i*stride] for i in range((len(cs)+1)//2)])
    for i in np.arange(len(centers)-2, -1, -1):
        cs = centers[i]
        idx = cs.index(cext)
        cext = cs[np.min([idx + ext, len(cs) - 1])]
    return cext

def plot_hm_ctrs(mf, bost, ilayer, itf, char, k, th, ff, channel):
    plt.figure(dpi=275)
    plt.imshow(mf, aspect='auto')
    plt.contour(bost, colors='red', levels=[0.5])
    plt.title('Network Dissection for %s Channel %d, L%d FM%d, k=%d' %(char, channel, ilayer, itf, k))
    plt.xlabel('Time')
    plt.ylabel('Muscles (Muscle / Spindle Firing Rate)')
    plt.savefig('%s/l%d/tf%d/%s/nwdiss_%d_th%s_ch%d.png' %(ff, ilayer, itf, char, k, th, channel))
    plt.savefig('%s/l%d/tf%d/%s/nwdiss_%d_th%s_ch%d.svg' %(ff, ilayer, itf, char, k, th, channel))
    plt.close()

# %% MAIN

def main(model, runinfo):

    ff = runinfo.analysisfolder(model, 'network_dissection')
    os.makedirs(ff, exist_ok=True)

    datafolder = runinfo.datafolder(model)

    #IMPORT DATA
    data, xyplmvt = X_data('mf', runinfo, datafolder, polar=False)
    labels, _ = X_data('labels', runinfo, datafolder, polar=False)

    layers = []

    nlayers = model['nlayers']
    for ilayer in np.arange(-1, nlayers):
        '''
        if ilayer==-1:
                layer='data'

        else:
            layer = 'l%d' %ilayer
        lo = pickle.load(open(datafolder + layer + '_10pc.pkl', 'rb'))
        '''
        lo = read_layer_reps(ilayer, runinfo, model)
        layers.append(lo[xyplmvt])

    # %% DISSECTION        
    #Store highest activations for every layer
    actmaps = []
    idxs = []

    for ilayer, layer in enumerate(layers):
        actmaps.append([])
        idxs.append([])
        for itf in range(layer.shape[3]):
            ams, ids = topk_actmaps(layer[...,itf], labels)
            actmaps[ilayer].append(ams)
            idxs[ilayer].append(ids)

    for ilayer in range(nlayers + 1):
        print("Layer %d" %ilayer)
        for itf in range(len(actmaps[ilayer])):
            print("TF %d" %itf)
            for ichar, char in enumerate(char_labels):
                try:
                    os.makedirs('%s/l%d/tf%d/%s/'%(ff, ilayer, itf, char))
                except:
                    print('folder already exists')
                
                for k in range(topk):
                    for channel in range(data.shape[-1]):
                        #print(k)
                        idx = idxs[ilayer][itf][ichar][k].astype(int)
            
                        mf = data[idx, ..., channel]
                        am = actmaps[ilayer][itf][ichar][k]
                        resized = resize(am, mf.shape)
                        bat = above_threshold(resized, threshold=0.5)
                        
                        plot_hm_ctrs(mf, bat, ilayer, itf, char, k, '05', ff, channel)
                        bat = above_threshold(resized, threshold=0.3)
                        plot_hm_ctrs(mf, bat, ilayer, itf, char, k, '03', ff, channel)
        plt.close('all')
