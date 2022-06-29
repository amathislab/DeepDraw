#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:43:21 2019

@author: kai

v2: Clean up code, set up functions, optimized alpha etc
v1: PCA and tSNE plots, looping
"""

import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

from rowwise_neuron_curves_controls import X_data

char_labels = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']
    

# %% TSNE

def mytsne(X):
    
    pca = PCA(n_components=50)
    #print(X.shape)
    pca_result = pca.fit_transform(X)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    print(f'Total explained variance: {sum(pca.explained_variance_ratio_)}')

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pca_result)
    
    df = dict()
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    
    return tsne, df

def plottsne(df, name, labels, ff, alpha=0.8):
    plt.figure(dpi=275)
    plt.scatter(
        x = df['tsne-2d-one'],
        y = df['tsne-2d-two'],
        #s=4,
        edgecolors = 'none',
        c=19 - labels,
        cmap='tab20_r',
        alpha = alpha
    )
    cb = plt.colorbar()
    cb.set_ticks(np.linspace(0.5, 19.5, 21))
    cb.set_ticklabels(char_labels[::-1])
    plt.title('%s tSNE' %name)
    plt.xlabel('tsne-pca50-one')
    plt.ylabel('tsne-pca50-two')
    #plt.savefig('%s%s_tsne%d.png' %(ff,name, int(alpha*100)))
    filename = os.path.join(ff, 'tsne_%s.png' %name)
    print('saving in file %s ...' %filename)
    plt.savefig(filename)
    plt.close()
    
def main(model, runinfo):
    nlayers = model['nlayers'] + 1
    
    labels, xyplmvt = X_data('labels', runinfo, datafolder=runinfo.datafolder(model))
    
    ff = runinfo.analysisfolder(model, 'tsne')
    
    nlayers= model['nlayers'] + 1
    
    os.makedirs(ff, exist_ok=True)
    
    # %% LOOP OVER ALL LAYERS
    
    for i in range(nlayers):
        lname = "l%d" %i
        
        print(lname)
        if i == 0:
            l = pickle.load(open(os.path.join(runinfo.datafolder(model), 'data.pkl'), 'rb'))
        else:
            l = pickle.load(open(os.path.join(runinfo.datafolder(model), 'l%d.pkl' %(i - 1)), 'rb'))
        l = l[xyplmvt]
        
        #print(l.shape)
        
        lx = l.reshape(l.shape[0], -1)
        
        try:
            ltsne, ldf = mytsne(lx)
            plottsne(ldf, lname, labels, ff)
        except ValueError:
            print('not enough samples to compute PCA & tSNE')