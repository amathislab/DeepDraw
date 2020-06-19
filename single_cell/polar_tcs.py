#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:26:47 2020

@author: kai
"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from rowwise_neuron_curves_controls import *
import pickle
import os
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

muscle_order = ['CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP', 'LAT1', 'LAT2', 'LAT3', 'PECM1', 
                'PECM2', 'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN', 'ANC', 'BIClong', 'BICshort', 
                'BRA', 'BRD', 'ECRL', 'PT', 'TRIlat', 'TRIlong', 'TRImed']

# %% POLAR TUNING CURVE PLOT
        
def polartc(thetas, rs, ilayer, k, irow, r2, expf, fset='vel'):
    """Plot polar directional tuning curve"""
    randsel = np.random.permutation(len(thetas))[:2000]

    fig = plt.figure(dpi=275)
    plt.polar(thetas[randsel], rs[randsel], 'x', alpha=0.7)
    plt.title(str(r2))
    
    plt.savefig('%s/%s_l%d_%s_polar_r2_bl.pdf' %(expf, fset, ilayer + 1, k))
    
def carttc(thetas, rs, ilayer, k):
    """Plot Cartesian directional tuning curve"""
    plt.figure(dpi=275)
    plt.scatter(thetas, rs)
    plt.title('Directional Tuning for Node L%d ' %(ilayer, str(idx)))
    plt.savefig('%s/l%d/%s_l%d_%s_cart.pdf' %(expf, ilayer + 1, fset, ilayer + 1, str(k)))
        
def get_thetas_rs_from_row(polar, acts, rowidx, tcoff=32):
    """Get matching thetas and activations (stored as Rs)    """
    
    print(rowidx, acts.shape)
    
    thetas = polar[:,1]

    centers = get_centers(acts.shape[2])
    
    if(len(rowidx) == 1):
        nodeselector = (slice(None),) + rowidx
    else:
        nodeselector = (slice(None),) + tuple([rowidx[0]]) + (slice(None),) + tuple([rowidx[1]])
    
    acts = acts[nodeselector]
    
    print(len(centers))
    print(thetas.shape)
    print(acts.shape)
    
    fmtcoff = sum(np.where(centers <= tcoff, True, False))
    
    thetas = thetas[:,centers[fmtcoff:len(centers) - fmtcoff]]
    acts = acts[:,fmtcoff:acts.shape[1]-fmtcoff]
    
    nnas = np.invert(np.isnan(thetas))
    
    thetas = thetas[nnas]
    rs = acts[nnas]
    
    if(len(rowidx)==1):
        rs = rs + 82
    
    return thetas, rs


# %% 3D PLOTS FOR DIR + VEL TUNING

def dirvelplotpolar(thetas, vms, acts, ilayer, k, rowidx, r2, expf, fset='vel'):
    fig = plt.figure(dpi=275)
    ax = fig.add_subplot(111, projection='polar')

    duplicate = np.where(thetas > np.pi/2)[0]
    thetasdpl = thetas[duplicate] - 2*np.pi
    vmsdpl = vms[duplicate]
    actsdpl = acts[duplicate]
    thetas = np.concatenate((thetas, thetasdpl))
    vms = np.concatenate((vms, vmsdpl))
    acts = np.concatenate((acts, actsdpl))
    
    ti = np.linspace(-3*np.pi/2, np.pi, 100)
    ri = np.linspace(0, vms.max(), 100)
    zi = griddata((thetas, vms), acts, (ti[None,:], ri[:, None]), method='linear')
    

    #cover non-reached points with value obtained by 'nearest'
    zinearest = griddata((thetas, vms), acts, (ti[None,:], ri[:, None]), method='nearest')
    isnan = np.isnan(zi[:,int(zi.shape[1]/20):int(zi.shape[1]*19/20)])

    zi[:,int(zi.shape[1]/20):int(zi.shape[1]*19/20)][isnan] = zinearest[:,int(zi.shape[1]/20):int(zi.shape[1]*19/20)][isnan]

    #smoothen with a gaussian filter
    kernelfactor = 0.5
    zi = gaussian_filter(zi, sigma=[2*np.pi*kernelfactor, vms.max()*kernelfactor])
    plt.contourf(ti, ri, zi)
        
    randsel = np.random.permutation(len(thetas))[:2000]
    thetas = thetas[randsel]
    vms = vms[randsel]
    plt.title(str(r2))
    plt.colorbar(pad=0.07)
    
    plt.savefig('%s/dirvel_l%d_%d_3dpolar_smoothen_0%d_rc_nn.pdf' %(expf, ilayer + 1, k, int(kernelfactor*100)))
    plt.close('all')
    
def get_thetas_vms_rs_from_row(polar, acts, rowidx, tcoff=32):
    """Get matching thetas and activations (stored as Rs)    """
    
    print(rowidx, acts.shape)
    
    #select proper row
    vms = polar[:,0]
    thetas = polar[:,1]

    centers = get_centers(acts.shape[2])
    
    if(len(rowidx) == 1):
        nodeselector = (slice(None),) + rowidx
    else:
        nodeselector = (slice(None),) + tuple([rowidx[0]]) + (slice(None),) + tuple([rowidx[1]])
    
    print(nodeselector)
    print(acts.shape)
    acts = acts[nodeselector]

    #select time interval
    
    fmtcoff = sum(np.where(centers <= tcoff, True, False))
    
    thetas = thetas[:,centers[fmtcoff:len(centers) - fmtcoff]]
    vms = vms[:, centers[fmtcoff:len(centers) - fmtcoff]]
    acts = acts[:,fmtcoff:acts.shape[1]-fmtcoff]
    
    nnas = np.invert(np.isnan(thetas))
    
    thetas = thetas[nnas]
    vms = vms[nnas]
    #rs = acts[nodeselector].squeeze()
    rs = acts[nnas]
    
    return thetas, vms, rs

# %% TRAINED MODEL
        
def main(model, runinfo):
    
    nlayers = model['nlayers'] + 1
    dirmi = []
    dvmi = [] #store max indices in nested list
    
    kbest = 5
    mmod = 'std'
    fset = 'vel'
    
    for ilayer in np.arange(0,nlayers):
        evals = np.load(os.path.join(runinfo.resultsfolder(model, 'vel'), 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, fset, mmod, runinfo.planestring())))
        r2 = evals[...,1]
        #mcboxplot(r2, 'r2 Score', ilayer);
        dirtuning = np.copy(r2[...,1])
        dirtuning[dirtuning == 1] = -1
        dvtuning = np.copy(r2[...,3])
        dvtuning[dirtuning == 1] = -1
        dirmaxids = np.argsort(- dirtuning.flatten())[:kbest]
        print("Best %d Nodes for Directional Tuning Layer %d Where r2 != 1:" %(kbest, ilayer))
        dirmi.append([])
        for idx in dirmaxids:
            ui = np.unravel_index(idx, dirtuning.shape)
            print(ui, 'r2 score: ', dirtuning[ui])
            dirmi[ilayer].append((ui, dirtuning[ui]))
        
        dvmaxids = np.argsort(- dvtuning.flatten())[:kbest]
        print("Best %d Nodes for Dir + Vel Tuning Layer %d Where r2 != 1:" %(kbest, ilayer))
        dvmi.append([])
        for idx in dvmaxids:
            ui = np.unravel_index(idx, dvtuning.shape)
            print(ui, 'r2 score: ', dvtuning[ui])
            dvmi[ilayer].append((ui, dvtuning[ui]))
            
    #print(runinfo)
    datafolder=runinfo.datafolder(model)
    polar, xyplmvt = X_data('vel', runinfo=runinfo, datafolder=datafolder)
    
    expf = runinfo.analysisfolder(model, 'polar_tcs')
    os.makedirs(expf, exist_ok = True)
    
    ###output 2D polar plots
    for ilayer in np.arange(-1, len(dirmi) - 1):
    
        print(ilayer+1)
            
        lo = pickle.load(open(os.path.join(datafolder, lstring(ilayer) + '.pkl'), 'rb'))
        lo = lo[xyplmvt]
        
        for k in range(kbest):         
            rowidx = dirmi[ilayer +1][k]
            thetas, rs = get_thetas_rs_from_row(polar, lo, rowidx[0])
            polartc(thetas, rs, ilayer, k, rowidx[0], rowidx[1], expf, fset=fset)
            plt.close('all')      
       
    for il in np.arange(-1, nlayers - 1):
        layer = lstring(il)
        lo = pickle.load(open(os.path.join(datafolder, lstring(il) + '.pkl'), 'rb'))
        lo = lo[xyplmvt]
        
        print(layer)
        for k in range(kbest):
            print(k)
            rowidx = dvmi[il + 1][k]
            thetas, vms, rs = get_thetas_vms_rs_from_row(polar, lo, rowidx[0])
            dirvelplotpolar(thetas, vms, rs, il, k, rowidx[0], rowidx[1], expf, fset=fset)
               