#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:43:21 2019

@author: kai
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import pickle, os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

datafolder = '../data/'
modelname = 'gru_2_32-32_256_low'
#modelname = 'lstm_4_8-32-64-64_512_med' #test
#modelname = 'lstm_4_32-32-64-64_256_low'
#modelname = 'fc_4_8-16-32-64_8-8-32-64_73_low'
#modelname = 'spatiotemporal_4_8-8-32-64_7292'
#modelname = 'temporal_spatial_4_16-16-32-64_64-64-64-64_5272'
#modelname = 'gru_2_32-32_256_low'
#modelname= 's_t_44_best'
sn = 'gru'
nlayers = 3
kinvars = pd.read_hdf(datafolder + modelname + '/kinvars_10pc.hdf5')
ee = kinvars['endeffector_coords'].values
ee = np.swapaxes(ee, 0, 1)

mf = pickle.load(open(datafolder + modelname + "/data_10pc.pkl", "rb"))
labels = pickle.load(open(datafolder + modelname + "/labels_10pc.pkl", "rb"))

char_labels = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']

#random_idx = np.random.permutation(len(mf))[:20]

mfx = mf.reshape(mf.shape[0], -1)
eex = ee.reshape(ee.shape[0], -1)

os.mkdir(f'{modelname}/')

cm = plt.get_cmap('tab20', 19)
# %% PCA
## MF
#FIRST INCLUDE PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(mfx)

df = dict()
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#plt.figure(figsize=(16,10))
plt.clf()
plt.scatter(
    x = df['pca-one'],
    y = df['pca-two'],
    c=labels,
    cmap='tab20',
)
cb = plt.colorbar()
cb.set_ticks(np.linspace(0.5, 19.5, 21))
cb.set_ticklabels(char_labels)
plt.title('Muscle Spindle Firing Rates per Label, Dim Red by PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig(f'{modelname}/mf_pca.png')

##EE
ee_pca = PCA(n_components=3)
ee_pca_result = ee_pca.fit_transform(eex)

ee_df = dict()
ee_df['pca-one'] = ee_pca_result[:,0]
ee_df['pca-two'] = ee_pca_result[:,1] 
ee_df['pca-three'] = ee_pca_result[:,2]

print('Explained variation per principal component: {}'.format(ee_pca.explained_variance_ratio_))

#plt.figure(figsize=(16,10))
plt.clf()
plt.scatter(
    x = ee_df['pca-one'],
    y = ee_df['pca-two'],
    c=labels,
    cmap=cm,
)
cb = plt.colorbar()
cb.set_ticks(np.linspace(0.5, 19.5, 21))
cb.set_ticklabels(char_labels)
plt.title('Endeffectors per Label, Dim Red by PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig(f'{modelname}/ee_pca.png')
    
# %% TSNE

##MF
mf50pca = PCA(n_components=50)
mf50pca_result = mf50pca.fit_transform(mfx)
print('Explained variation per principal component: {}'.format(mf50pca.explained_variance_ratio_))
print(f'Total explained variance: {sum(mf50pca.explained_variance_ratio_)}')

mf_tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
mf_tsne_results = mf_tsne.fit_transform(mf50pca_result)

mf50df = dict()
mf50df['tsne-2d-one'] = mf_tsne_results[:,0]
mf50df['tsne-2d-two'] = mf_tsne_results[:,1]

#plt.figure(figsize=(16,10))
plt.clf()
plt.scatter(
    x = mf50df['tsne-2d-one'],
    y = mf50df['tsne-2d-two'],
    c=labels,
    cmap='tab20',
)
cb = plt.colorbar()
cb.set_ticks(np.linspace(0.5, 19.5, 21))
cb.set_ticklabels(char_labels)
plt.title('Muscle Spindle Firing Rates Output per Label, Dim Red by PCA + TSNE')
plt.xlabel('tsne-pca50-one')
plt.ylabel('tsne-pca50-two')
plt.savefig(f'{modelname}/mf_tsne.png')

##EE
ee50pca = PCA(n_components=50)
ee50pca_res = ee50pca.fit_transform(eex)
print('Explained variation per principal component: {}'.format(ee50pca.explained_variance_ratio_))
print(f'Total explained variance: {sum(mf50pca.explained_variance_ratio_)}')

ee_tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
ee_tsne_res = mf_tsne.fit_transform(ee50pca_res)

ee50d = dict()
ee50d['tsne-2d-one'] = ee_tsne_res[:,0]
ee50d['tsne-2d-two'] = ee_tsne_res[:,1]

#plt.figure(figsize=(16,10))
plt.clf()
plt.scatter(
    x = ee50d['tsne-2d-one'],
    y = ee50d['tsne-2d-two'],
    c=labels,
    cmap='tab20',
)
cb = plt.colorbar()
cb.set_ticks(np.linspace(0.5, 19.5, 21))
cb.set_ticklabels(char_labels)
plt.title('Endeffector Coordinates Output per Label, Dim Red by PCA + TSNE')
plt.xlabel('tsne-pca50-one')
plt.ylabel('tsne-pca50-two')
plt.savefig(f'{modelname}/ee_tsne.png')
    
# %% LAYERS
    
for i in range(nlayers):
    print("Layer %d"%i)
    l = pickle.load(open(datafolder + f"{modelname}/l{i}_10pc.pkl", "rb"))
    lx = l.reshape(l.shape[0], -1)

    ##L8
    l8_pca = PCA(n_components=3)
    l8_pca_result = l8_pca.fit_transform(lx)
    
    l8_df = dict()
    l8_df['pca-one'] = l8_pca_result[:,0]
    l8_df['pca-two'] = l8_pca_result[:,1] 
    l8_df['pca-three'] = l8_pca_result[:,2]
    
    print('Explained variation per principal component: {}'.format(l8_pca.explained_variance_ratio_))
    
    #plt.figure(figsize=(16,10))
    plt.clf()
    plt.scatter(
        x = l8_df['pca-one'],
        y = l8_df['pca-two'],
        c=labels,
        cmap='tab20',
    )
    cb = plt.colorbar()
    cb.set_ticks(np.linspace(0.5, 19.5, 21))
    cb.set_ticklabels(char_labels)
    plt.title(f'Layer {i+1} Output per Label, Dim Red by PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.savefig(f'{modelname}/{sn}_l{i+1}_pca.png')
    
    ##L8 OUTPUT
    l850pca = PCA(n_components=50)
    l850pca_res = l850pca.fit_transform(lx)
    print('Explained variation per principal component: {}'.format(l850pca.explained_variance_ratio_))
    print(f'Total explained variance: {sum(l850pca.explained_variance_ratio_)}')
    
    l8_tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    l8_tsne_res = mf_tsne.fit_transform(l850pca_res)
    
    l850d = dict()
    l850d['tsne-2d-one'] = l8_tsne_res[:,0]
    l850d['tsne-2d-two'] = l8_tsne_res[:,1]
    
    #plt.figure(figsize=(16,10))
    plt.clf()
    plt.scatter(
        x = l850d['tsne-2d-one'],
        y = l850d['tsne-2d-two'],
        c=labels,
        cmap='tab20',
    )
    cb = plt.colorbar()
    cb.set_ticks(np.linspace(0.5, 19.5, 21))
    cb.set_ticklabels(char_labels)
    plt.title(f'Layer {i+1} Output per Label, Dim Red by PCA + TSNE')
    plt.xlabel('tsne-pca50-one')
    plt.ylabel('tsne-pca50-two')
    plt.savefig(f'{modelname}/{sn}_l{i+1}_tsne.png')