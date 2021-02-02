#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:20:57 2019
Majorly revamped on Sun Jan 31 2021

@author: kai
"""

import numpy as np
import pandas as pd
#from sklearn import linear_model
#from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
#matplotlib.use('default')
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#from scipy import exp
import pickle
import os
#from scipy.misc import imresize
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
    return np.max(actmap.flatten())

def topk_actmaps(actmaps, labels):
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
    #if fmapntime != len(centers):
    for i in range(int(np.log2(len(centers)/ fmapnt))):
        centers = [centers[i*t_stride] for i in range(len(centers)//2)]
    
    assert len(centers) == fmapnt, "length of centers and fmapnt not equal!"
    
    return centers[fmapt]

def get_s(fmaps, fmapns, ns = 25, s_stride = 2):
    """Converts time dimension index from feature map to original time point"""
    centers = np.arange(ns)
    #print("fmapns: %d" %fmapns)
    #if fmapntime != len(centers):
    while len(centers) > fmapns:
        centers = [centers[i*t_stride] for i in range((len(centers)+1)//2)]
    #print(centers)
    
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
        #print(i, idx)
        cext = cs[np.min([idx + ext, len(cs) - 1])]
    return cext

def translate(bat, osh = [25, 320]):
    """Returns boolean map in original space time"""
    bost = np.zeros(osh).astype(bool)
    for idx, b in np.ndenumerate(bat):
        if b:
            s, t = idx[0], idx[1]
            sext, text = skernelsize // 2, tkernelsize // 2
            #print(s, t)
            if bat.shape[0] < osh[0]:
                s = get_s(idx[0], bat.shape[0], osh[0], s_stride)
                sext = get_ext(sext, bat.shape[0], nmuscles, s_stride)
            if bat.shape[1] < osh[1]:
                t = get_t(idx[1], bat.shape[1], osh[1], t_stride)
                text = get_ext(text, bat.shape[1], ntime, t_stride)                    
            smin = np.max([0, s - sext])
            smax = np.min([osh[0], s + sext + 1])
            tmin = np.max([0, t - text])
            tmax = np.min([osh[1], t + text + 1])
            for sidx in np.arange(smin, smax):
                for tidx in np.arange(tmin, tmax):
                    bost[sidx, tidx] = True
    return bost

def plot_hm_ctrs(mf, bost, ilayer, itf, char, k, th, ff, channel):
    plt.figure(dpi=275)
    plt.imshow(mf, aspect='auto')
    plt.contour(bost, colors='red', levels=[0.5])
    plt.title('Network Dissection for %s Channel %d, L%d FM%d, k=%d' %(char, channel, ilayer, itf, k))
    plt.xlabel('Time')
    plt.ylabel('Muscles (Muscle / Spindle Firing Rate)')
    plt.savefig('%s/l%d/tf%d/%s/nwdiss_%d_th%s_ch%d.png' %(ff, ilayer, itf, char, k, th, channel))
    plt.close()
'''
am = actmaps[0][0][0,0]
idx = idxs[0][0][0,0].astype(int)

mf = data[idx]

resized = resize(am, mf.shape)
bat = above_threshold(resized, threshold=0.5)

#bost = translate(bat)


plot_hm_ctrs(mf, bat, 0, 0 , 'a', 0)

'''

# %% MAIN

def main(model, runinfo):

    ff = runinfo.analysisfolder(model, 'network_dissection')
    os.makedirs(ff, exist_ok=True)

    datafolder = runinfo.datafolder(model)

    #IMPORT DATA
    #kinvars = pd.read_hdf(datafolder + 'kinvars_10pc.hdf5')    
    #mc = np.swapaxes(kinvars['muscle_coords'].values, 0, 1)
    #labels = pickle.load(open(datafolder + 'labels_10pc.pkl', 'rb'))
    #data = pickle.load(open(datafolder + 'data_10pc.pkl', 'rb'))
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
                #print(char)
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
                        #bost = translate(bat)
                        
                        plot_hm_ctrs(mf, bat, ilayer, itf, char, k, '05', ff, channel)
                        bat = above_threshold(resized, threshold=0.3)
                        plot_hm_ctrs(mf, bat, ilayer, itf, char, k, '03', ff, channel)
        plt.close('all')
