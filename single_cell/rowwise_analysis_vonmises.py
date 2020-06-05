#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:57:21 2019

@auhor: kai

v5: introduced smoothing, baseline offset for muscle spindles, fixed nearest neighbor analysis edge effects
v4: implemented axis lower limit on polar plots and r2 score labels
v3: implemented subsampling and alpha for tuning curves,
    performed interpolation for contours in cartesian space (dir + vel tuning curves)
v2: separate training and test set
"""

#from pyglmnet import GLM
import numpy as np
import matplotlib
matplotlib.use('agg')
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from dir_tuning_alllayers_mp import *
from rowwise_neuron_curves_vonmises import *
import pickle
import os
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from vonmifit import *

# %% PARS AND FIRST OVERVIEW

# GLOBAL PARS
modelname = 'spatial_temporal_4_8-16-16-32_64-64-64-64_5272'
#modelname = 'temporal_spatial_4_16-16-32-64_64-64-64-64_5272'
#modelname = 'spatiotemporal_4_8-8-32-64_7292'
datafolder = '../data/%s/' %modelname
nlayers = 9
#nlayers += 1
#t_kernelsize = 7
t_stride = 2
ntime=320
metrics = ['RMSE', 'r2', 'PCC']
nmetrics = len(metrics)
expid = 2
fset = 'vel'
mmod = 'vonmises'
expf= '%s/%s/exp%d/' %(modelname, fset, expid)
ff = 'figs/%s' %expf
tcoff = 32

ilayer = -1

evals = np.load('%s/%s/exp%d/l%d_%s_mets_%s_test.npy' %(modelname, fset, expid, ilayer+1, fset, mmod))        

#Overview metrics
print(evals[...,1].max())
print(evals[...,1].mean())
print(evals[...,1].min())

# %% DATA READ IN 
   
alltestevals = []
alltrainevals = []

os.makedirs(ff, exist_ok = True)

for ilayer in np.arange(0,nlayers):
    testevals = np.load('%sl%d_%s_mets_%s_test.npy' %(expf, ilayer, fset, mmod))        
    alltestevals.append(testevals)
    
    trainevals = np.load('%sl%d_%s_mets_%s_train.npy' %(expf, ilayer, fset, mmod))        
    alltrainevals.append(trainevals)

# %% SINGLE BOXPLOT ACROSS ALL LAYERS
        
def allbp(allevals, dataset = 'test'):
    
    tctypes = ['Base LinReg', 'Direction', 'Velocity', 'Vel + Dir']
    color = ['blue', 'red', 'green', 'orange']
    
    #plot parameters
    
    nmods = 4
    space = 0.75
    width = 0.6
    lspace = space*nmods + 1
    
    #plot figure
    plt.figure(figsize=(14,6), dpi=200)
    bp = []
    for ilayer in np.arange(0,nlayers):
        for i in range(nmods):
            r2 = allevals[ilayer][...,i,1].flatten()
            r2 = r2[r2 != 1]
            bp.append(plt.boxplot(r2, notch=False, 
                        positions=[ilayer*lspace+0.75*i+1], 
                        widths=width, patch_artist = True))
    
    for i, p in enumerate(bp):
        plt.setp(p['boxes'], color=color[i%nmods])
    
    plt.title('r2 Scores for Velocity and Direction Models')
    
    plt.xticks(np.arange(lspace/2,8*nmods+lspace/2+1,lspace), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,nlayers+1)])
                   #horizontalalignment='left')
    plt.ylabel('r2 Scores')
    plt.ylim(-0.1, 1.0)
    plt.xlim(0, nlayers * lspace + 1)
    plt.legend([bp[i]['boxes'][0] for i in range(nmods)], tctypes, loc='upper right')
    plt.savefig('figs/%s/%s_bpal_%s.png' %(expf, fset, dataset))
    
    print("Plot saved.")
    plt.close('all')
    
allbp(alltestevals, 'test')
allbp(alltrainevals, 'train')


# %% DATA READIN FOR PLOTS

polar, xyplmvt = X_data()

def get_thetas_rs_from_row(polar, acts, rowidx, tcoff=32):
    """Get matching thetas and activations (stored as Rs)    """
    
    print(rowidx, acts.shape)
    
    #select proper row
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
    #select time interval
    
    fmtcoff = sum(np.where(centers <= tcoff, True, False))
    
    thetas = thetas[:,centers[fmtcoff:len(centers) - fmtcoff]]
    acts = acts[:,fmtcoff:acts.shape[1]-fmtcoff]
    
    nnas = np.invert(np.isnan(thetas))
    
    thetas = thetas[nnas]
    #rs = acts[nodeselector].squeeze()
    rs = acts[nnas]
    
    #in case of muscle spindles, introduce baseline offset
    if(len(rowidx)==1):
        rs = rs + 82
    
    return thetas, rs

# %% BOXPLOTS AND EXTRACT MAX INDICES

dirmi = []
dvmi = [] #store max indices in nested list

kbest = 5

for ilayer in np.arange(0,nlayers):
    evals = np.load('%sl%d_%s_mets_%s_test.npy' %(expf, ilayer, fset, mmod))        
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

# %% POLAR TUNING CURVE PLOT
        
def polartc(thetas, rs, ilayer, k, irow, r2):
    """Plot polar directional tuning curve"""
    randsel = np.random.permutation(len(thetas))[:2000]

    fig = plt.figure(dpi=275)
    plt.polar(thetas[randsel], rs[randsel], 'x', alpha=0.7)
    #ax = plt.gca()
    plt.title('Directional Tuning L%d Row %s , r2=%f' %(ilayer + 1, str(irow), r2), pad=17)
    #ax.set_ylim(bottom = 0)
    
    plt.savefig('figs/%s/l%d/%s_l%d_%s_polar_r2_bl.png' %(expf, ilayer + 1, fset, ilayer + 1, k))
    
def carttc(thetas, rs, ilayer, k):
    """Plot Cartesian directional tuning curve"""
    plt.figure(dpi=275)
    plt.scatter(thetas, rs)
    plt.title('Directional Tuning for Node L%d ' %(ilayer, str(idx)))
    plt.savefig('figs/%s/l%d/%s_l%d_%s_cart.png' %(expf, ilayer + 1, fset, ilayer + 1, str(k)))

for ilayer in np.arange(-1, len(dirmi) - 1):
    
    print(ilayer+1)
        
    lo = pickle.load(open(datafolder + lstring(ilayer) + '_10pc.pkl', 'rb'))
    lo = lo[xyplmvt]
    
    for k in range(kbest):         
        #exclude constants
        try:
            os.makedirs('figs/%sl%d' %(expf, ilayer + 1) )
        except:
            print('folder already exists')
        rowidx = dirmi[ilayer +1][k]
        thetas, rs = get_thetas_rs_from_row(polar, lo, rowidx[0])
        polartc(thetas, rs, ilayer, k, rowidx[0], rowidx[1])
        #carttc(thetas, rs, ilayer, k)
        plt.close('all')

# %% POLAR PLOTS OVERLAID WITH FITTED VAN MISES FUNCTION

def polarsctc(thetas, rs, ilayer, idx, r2, r0, rmax, kappa, thetapd):
    """Plot polar directional tuning curve"""
    randsel = np.random.permutation(len(thetas))[:2000]

    fig = plt.figure(dpi=275)
    plt.polar(thetas[randsel], rs[randsel], 'x', alpha=0.7)
    #ax = plt.gca()
    #ax.set_ylim(bottom = 0)
    
    linth = np.linspace(-np.pi, np.pi, 200)
    ftc = von_mises(linth, r0, rmax, kappa, thetapd)
    if(len(idx) == 1):
        ftc = ftc + 82
    else:
        plt.ylim((0, rs.max()))

    
    plt.polar(linth, ftc, color='r', linestyle='-', linewidth=2)
    
    parstr = 'Parameters: \n r0: %.4f \n rmax: %.4f \n kappa: %.4f \n thetapd: %.4f' %(r0, rmax, kappa, thetapd)
    
    fig.text(0.76, 0.7, parstr, bbox=dict(fc="none"))
    
    plt.title('Directional Tuning L%d Row %s \n With Fitted Von Mises Curve , r2=%f' %(ilayer, str(idx), r2), pad=17)
    plt.tight_layout()
    plt.savefig(os.path.join(ff, 'sctc', 'l%d' %ilayer, 'vonmi_l%d_n%s_polar_sctc.png' %(ilayer, str(idx))))

for ilayer, testevals in enumerate(alltestevals):
    lo = pickle.load(open(datafolder + lstring(ilayer -1) + '_10pc.pkl', 'rb'))
    lo = lo[xyplmvt]
    os.makedirs(os.path.join(ff, 'sctc', 'l%d'%ilayer), exist_ok = True)
    for idx in np.ndindex(testevals.shape[:-2]):
        thetas, rs = get_thetas_rs_from_row(polar, lo, idx)
        r2 = testevals[idx][1,1]
        pars=testevals[idx][1,3:]
        polarsctc(thetas, rs, ilayer, idx, r2, *pars)

# %% POLAR PLOT TESTING
        
def polartc(thetas, rs, ilayer, k):
    """Plot polar directional tuning curve"""
    randsel = np.random.permutation(len(thetas))[:500]
    plt.figure(dpi=275)
    plt.polar(thetas[randsel], rs[randsel], 'x', alpha = 1)
    plt.title('Directional Tuning Curve Showing Activity for L%d Row %s ' %(ilayer, str(idx)), pad=17)
    
    plt.savefig('figs/%s/l%d/%s_l%d_%s_polar_test7.png' %(expf, ilayer + 1, fset, ilayer + 1, str(k)))
    
polartc(thetas, rs, ilayer, k)

# %% 3D PLOTS FOR DIR + VEL TUNING

"""
def dirvelplot(thetas, vms, acts, ilayer, idx, k):
    fig = plt.figure(dpi=275)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(thetas, vms, acts)
    
    plt.title("Direction and Velocity Tuning Curve for L%d %s" %(ilayer + 1, str(idx)))
    ax.set_xlabel('Theta Direction [rads]')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Activation')
    
    plt.savefig('figs/%s/l%d/dirvel_l%d_%d_3d.png' %(expf, ilayer + 1, ilayer + 1, k))
"""

def dirvelplotpolar(thetas, vms, acts, ilayer, k, rowidx, r2):
    fig = plt.figure(dpi=275)
    ax = fig.add_subplot(111, projection='polar')
    #ax = fig.add_subplot(111)
    #cntr = ax.tricontourf(thetas, vms, acts, levels=20)
    
    #duplicate certain regions in order to cover break at np.pi

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
    #isnan = np.isnan(zi)
    isnan = np.isnan(zi[:,int(zi.shape[1]/20):int(zi.shape[1]*19/20)])
    #isnan = np.isnan(zi[:,int(len(zi)/5):])
    #zi[isnan] = zinearest[isnan]
    #zi[:,int(len(zi)/):][isnan] = zinearest[:,int(len(zi)/7):][isnan]
    zi[:,int(zi.shape[1]/20):int(zi.shape[1]*19/20)][isnan] = zinearest[:,int(zi.shape[1]/20):int(zi.shape[1]*19/20)][isnan]
    print(zi.shape, isnan.shape)
    #smoothen with a gaussian filter
        
    
    kernelfactor = 0.5
    zi = gaussian_filter(zi, sigma=[2*np.pi*kernelfactor, vms.max()*kernelfactor])

    '''
    xs = vms * np.cos(thetas)
    ys = vms * np.sin(thetas)
    
    #ti = np.linspace(-3*np.pi/2, np.pi, 100)
    #ri = np.linspace(0, vms.max(), 100)
    xi = (xs.min(), xs.max(), 1000)
    yi = (ys.min(), ys.max(), 1000)
    zi = griddata((xs, ys), acts, (xi[None,:], yi[:, None]), method='linear')
    
    #cover non-reached points with value obtained by 'nearest'
    zi = griddata((xs, ys), acts, (xi[None,:], yi[:, None]), method='nearest')
    isnan = np.isnan(zi)
    zi[isnan] = zinearest[isnan]
    '''
    
    #create contours and plot
    
    #plt.xlim(-np.pi, np.pi)
    #plt.ylim(0,0.16)
    plt.contourf(ti, ri, zi)
        
    randsel = np.random.permutation(len(thetas))[:2000]
    thetas = thetas[randsel]
    vms = vms[randsel]
    
    #ax.scatter(thetas, vms, c='black', s=3, alpha=0.6)
    #ax.axis('tight')
    
    plt.title("Direction x Velocity Tuning for L%d %s, r2=%f" %(ilayer + 1, str(rowidx), r2), pad=17)
    #ax.set_xlabel('Theta Direction [rads]')
    #ax.set_ylabel('Velocity')
    #ax.set_zlabel('Activation')
    
    plt.colorbar(pad=0.07)
    
    plt.savefig('figs/%s/l%d/dirvel_l%d_%d_3dpolar_smoothen_0%d_rc_nn.png' %(expf, ilayer + 1, ilayer + 1, k, int(kernelfactor*100)))
    plt.close('all')

    
def get_thetas_vms_rs_from_row(polar, acts, rowidx):
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

for il in np.arange(-1, nlayers):
    layer = lstring(il)
    lo = pickle.load(open(datafolder + layer + '_10pc.pkl', 'rb'))
    lo = lo[xyplmvt]
    
    print(layer)
    
    try:
        os.makedirs('figs/%s/l%d' %(expf, il + 1))
    except:
        print('folder already exists')
    
    for k in range(kbest):
        print(k)
        rowidx = dvmi[il + 1][k]
        thetas, vms, rs = get_thetas_vms_rs_from_row(polar, lo, rowidx[0])
        #dirvelplot(thetas, vms, rs, il, idx, k)
        dirvelplotpolar(thetas, vms, rs, il, k, rowidx[0], rowidx[1])

# %% TEST DIR VEL POLAR CONTOUR PLOTS
def dirvelplotpolar(thetas, vms, acts, ilayer, rowidx, k):
    fig = plt.figure(dpi=275)
    ax = fig.add_subplot(111, projection='polar')
    #cntr = ax.tricontourf(thetas, vms, acts, levels=20)
    
    #duplicate certain regions in order to cover break at np.pi
    duplicate = np.where(thetas > np.pi/2)[0]
    thetasdpl = thetas[duplicate] - 2*np.pi
    vmsdpl = vms[duplicate]
    actsdpl = acts[duplicate]
    thetas = np.concatenate((thetas, thetasdpl))
    vms = np.concatenate((vms, vmsdpl))
    acts = np.concatenate((acts, actsdpl))
    
    ti = np.linspace(-3*np.pi/2, np.pi, 1000)
    ri = np.linspace(0, vms.max(), 100)
    zi = griddata((thetas, vms), acts, (ti[None,:], ri[:, None]), method='linear')
    
    #cover non-reached points with value obtained by 'nearest'
    #zinearest = griddata((thetas, vms), acts, (ti[None,:], ri[:, None]), method='nearest')
    #isnan = np.isnan(zi)
    #zi[isnan] = zinearest[isnan]
    
    #create contours and plot
    
    #plt.xlim(-np.pi, np.pi)
    #plt.ylim(0,0.16)
    plt.contourf(ti, ri, zi)
    #ax.scatter(thetas, vms, c='black', s=3, alpha=0.1)
    #ax.axis('tight')
    
    plt.title("Polar Direction and Velocity Tuning Curve for L%d %s" %(ilayer + 1, str(rowidx)), pad=17)
    #ax.set_xlabel('Theta Direction [rads]')
    #ax.set_ylabel('Velocity')
    #ax.set_zlabel('Activation')
    
    plt.colorbar()
    
    plt.savefig('figs/%s/l%d/dirvel_l%d_%d_3dpolar_test6.png' %(expf, ilayer + 1, ilayer + 1, k))
    plt.close('all')

dirvelplotpolar(thetas, vms, rs, il, idx, k)

# %% TRAIN VS. TEST SCORES
flattenedtestevals = np.concatenate(tuple([testeval[...,1].flatten() for testeval in alltestevals]))
flattenedtrainevals = np.concatenate(tuple([traineval[...,1].flatten() for traineval in alltrainevals]))
plt.figure(dpi=250)
plt.scatter(flattenedtrainevals, flattenedtestevals)
plt.title("Train and Test r2 Scores for Dir & Vel Models")
plt.xlabel('Train r2 Score')
plt.ylabel('Test r2 Score')
plt.xlim((-50,1.1))
plt.ylim((-50,1.1))
plt.savefig(ff + 'traintestcomp.png')
