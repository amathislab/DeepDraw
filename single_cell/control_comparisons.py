#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:31:24 2019

@author: kai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:23:52 2019

@author: kai

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rowwise_neuron_curves_controls import *
import pickle
import os
from scipy.interpolate import griddata
from scipy.stats import ks_2samp, f, t, ttest_rel
import pycircstat
from corrstats import independent_corr
import astropy.stats
from matplotlib.ticker import FormatStrFormatter

import time

# %% PARS AND FIRST OVERVIEW
t_stride = 2
ntime=320
metrics = ['RMSE', 'r2', 'PCC']
nmetrics = len(metrics)
fset = 'vel'
mmod = 'std'
tcoff = 32

tcnames = ['dir', 'vel', 'dirvel', 'acc', 'labels', 'ee', 'eepolar']
uniquezs = list(np.array([-45., -42., -39., -36., -33., -30., -27., -24., -21., -18., -15.,
                     -12.,  -9.,  -6.,  -3.,   0.,   3.,   6.,   9.,  12.,  15.,  18.,
                     21.,  24.,  27.,  30.]).astype(int))
uniquexs = list(np.array([ 6.,  9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42.,
                     45., 48., 51., 54.]).astype(int))
uniqueheights = [uniquezs, uniquexs]
orientations = ['hor', 'vert']

corrtypes = ['pd corr', 'r2 corr']

compors = ['hors vs. hors', 'verts vs. verts', 'hor vs. verts', 'vert vs. hors']

def format_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)
    
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# %% rcParams and Settings

params = {
   'axes.labelsize': 16,
   'legend.fontsize': 12,
   'xtick.labelsize': 16,
   'ytick.labelsize': 12,
   'text.usetex': False,
   'figure.figsize': [8,8 ],
   'font.size': 20,
   'axes.titlepad': 20,
   'xtick.major.size': 4.5,
   'xtick.minor.size': 3.5,
   'xtick.major.width': 1,
   'ytick.major.size': 4.5,
   'ytick.major.width': 1,
   'ytick.minor.size': 3.5
   }

plt.rcParams.update(params)

# %% UTILS

def pv_to_sl_code(pv):
    """ Return significance level corresponding to p-value
    
    Arguments
    ---------
    pv : float, p-value
    
    Returns
    sl : int, significance level (0-3)
    
    """
    sl = 0
    if(pv < 0.1):
        sl = 1
        if(pv < 0.05):
            sl = 2
            if(pv < 0.01):
                sl = 3
    return sl

def get_pds_sp(model, runinfo, r2threshold =  None):
    """ Get pds in a single plane 
    
    Arguments
    ---------
    model : dict
    runinfo : RunInfo (extension of dict)
    r2threshold : float, threshold above which test scores mean a neuron is directionally tuned

    Returns
    -------
    pds : list [nr_layers] of np.array [height index for planes, nr_neurons]
    """
    
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    
    fset = 'vel'
    mmod = 'std'
    
    
    ##SAVE PDs & R2s
    pds = []
    for ilayer in range(nlayers):
        
        resultsfolder = runinfo.resultsfolder(model)
        expf = '%s%s/' %(resultsfolder, 'vel')
        
        testevals = np.load('%sl%d_%s_mets_%s_%s_test.npy' %(expf, ilayer, fset, mmod, runinfo.planestring()))        
        if r2threshold is not None:
            testevals = testevals[testevals[...,1,1] > r2threshold] 
        if testevals.size > 0:
            dirtuning = testevals[...,1,3:5].reshape((-1,2))            
            prefdirs = np.apply_along_axis(angle_xaxis, 1, dirtuning)
        else:
            prefdirs = []
        """Create numpy for pref dirs with structure:
            Ax 1: Height Index
            Ax 2: Neurons
        """
        
        pds.append(prefdirs)
        
    return pds

def convert_str_to_float(astr):
    try:
        afl = float(astr)
    except:
        print('could not convert %s to float, set to 1 instead' %astr)
        afl = 1
    return afl

def sl_to_string(slgreater, sllesser = None):
    slstring = ''
    if(sllesser == None):
        for i in range(slgreater):
            slstring = slstring + '*'
    else:
        for i in range(slgreater):
            slstring = slstring + '+'
        for i in range(sllesser):
            slstring = slstring + '-'
    return slstring


# %% COMPILE PANDAS

def compile_comparisons_df(model, runinfo):
    #f.write("TC\tMean\tMedian\tStd\tMax\tMin\t90% Quantile\t10% Quantile\n")
    
    nlayers = model['nlayers'] + 1
    
    colnames = ['mean', 'median', 'std', 'max', 'min', 'q90', 'q10', 'ee', 'eepolar']
    #modelnames = [model['name']] + [model['name'] + '_%d' %(i + 1) for i in range(5)]
    modelbase = model['base']
    
    trainednamer = lambda i: modelbase + '_%d' %i
    controlnamer = lambda i: modelbase + '_%dr' %i
    modelnames = [namer(i) for i in np.arange(1,6) for namer in (trainednamer, controlnamer)]
    
    index = pd.MultiIndex.from_product((
                modelnames,
                list(range(nlayers)),
                tcnames),
                names = ('model', 'layer', 'tc'))
    df = pd.DataFrame(index=index, columns=colnames)
    
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    labelstats_index = [trainednames]  #for label 8 auROC score
    labelstats_df = pd.DataFrame(index=labelstats_index, columns=['median', 'max', 'median auROC', 'max auROC', 'n'])    
    
    
    eestats_columns = pd.MultiIndex.from_product((
            ['ee', 'eepolar'],
            ['median','max','q90','n']
            ), names=['tcname', 'metric'])
    print(modelnames)
    eestats_index = pd.MultiIndex.from_product((
            modelnames,
            ['L%d' %i for i in range(nlayers)]),
            names = ('model', 'layer'))
    eestats_df = pd.DataFrame(index=eestats_index, columns = eestats_columns)
    
    for im, mname in enumerate(modelnames):
        
        model_to_analyse = model.copy()
        model_to_analyse['name'] = mname
        #resultsfolder = runinfo.resultsfolder(model_to_analyse)
        
        expf={
              'vel': runinfo.resultsfolder(model_to_analyse, 'vel'),
              'acc': runinfo.resultsfolder(model_to_analyse, 'acc'),
              'labels': runinfo.resultsfolder(model_to_analyse, 'labels'),
              'ee': runinfo.resultsfolder(model_to_analyse, 'ee')
        }
        
        for ilayer in np.arange(0,nlayers):
            
            dvevals = np.load(os.path.join(expf['vel'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'vel', mmod, runinfo.planestring())))
            accevals = np.load(os.path.join(expf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std', runinfo.planestring())))
            labevals = np.load(os.path.join(expf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
            eeevals = np.load(os.path.join(expf['ee'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'ee', 'std', runinfo.planestring())))
            
            layerevals = []
            layerevals.append(dvevals[...,1,1]) #dir
            layerevals.append(dvevals[...,2,1]) #vel
            layerevals.append(dvevals[...,3,1]) #dir + vel
            layerevals.append(accevals[...,2,1]) #acc
            layerevals.append(labevals[...,0]) #labels
            layerevals.append(eeevals[...,0,1]) #ee
            layerevals.append(eeevals[...,3,1]) #eepolar
            
            
            print(eeevals[...,0,1])
            print(eeevals[...,3,1])
            
            
            for j, tcname in enumerate(tcnames):
                df.loc[(mname, ilayer, tcname), 'mean'] = layerevals[j].mean()
                df.loc[(mname, ilayer, tcname), 'median'] = np.median(layerevals[j])
                df.loc[(mname, ilayer, tcname), 'std'] = layerevals[j].mean()               
                df.loc[(mname, ilayer, tcname), 'max'] = layerevals[j].max()                
                df.loc[(mname, ilayer, tcname), 'min'] = layerevals[j].min()                
                df.loc[(mname, ilayer, tcname), 'q90'] = np.quantile(layerevals[j], 0.9)
                df.loc[(mname, ilayer, tcname), 'q10'] = np.quantile(layerevals[j], 0.10)
                
                
                if(ilayer == nlayers - 1 and tcname == 'labels'):
                    #print('recording label stats')
                    labelscores = layerevals[j]
                    labelstats_df.loc[mname, 'median'] = np.median(labelscores)
                    labelstats_df.loc[mname, 'max'] = labelscores.max()
                    labelstats_df.loc[mname, 'median auROC'] = np.median(labelscores/2 + 0.5)
                    labelstats_df.loc[mname, 'max auROC'] = (labelscores/2 + 0.5).max()              
                    labelstats_df.loc[mname, 'n'] = labelscores.size
                    
                if(tcname == 'ee'):
                    eescores = layerevals[j].reshape((-1,))
                    eestats_df.loc[(mname, 'L%d' %ilayer), ('ee', 'median')] = np.median(eescores)
                    eestats_df.loc[(mname, 'L%d' %ilayer), ('ee', 'max')] = eescores.max()             
                    eestats_df.loc[(mname, 'L%d' %ilayer), ('ee', 'q90')] = np.quantile(eescores, 0.9)            
                    eestats_df.loc[(mname, 'L%d' %ilayer), ('ee', 'n')] = eescores.size
                    
                if(tcname == 'eepolar'):
                    #print('recording label stats')
                    eescores = layerevals[j].reshape((-1,))
                    eestats_df.loc[(mname, 'L%d' %ilayer), ('eepolar', 'median')] = np.median(eescores)
                    eestats_df.loc[(mname, 'L%d' %ilayer), ('eepolar', 'max')] = eescores.max()     
                    eestats_df.loc[(mname, 'L%d' %ilayer), ('eepolar', 'q90')] = np.quantile(eescores, 0.9)             
                    eestats_df.loc[(mname, 'L%d' %ilayer), ('eepolar', 'n')] = eescores.size
    
    analysisfolder = runinfo.sharedanalysisfolder(model, 'kindiffs')
    os.makedirs(analysisfolder, exist_ok=True)
    df.to_csv(os.path.join(analysisfolder, model['base'] + '_comparisons_df.csv'))
    
    labelstats_df.to_csv(os.path.join(analysisfolder, model['base'] + '_labelstats_df.csv'))
    
    eestats_df.to_csv(os.path.join(analysisfolder, model['base'] + '_eestats_df.csv'))
    
    return df

def pairedt_quantiles(df, model, runinfo):
    
    idx = pd.IndexSlice
    
    nlayers = model['nlayers'] +1
    
    layers = list(range(nlayers))
    index = pd.MultiIndex.from_product((layers, tcnames))
    columns = ['pv']
    pt_df = pd.DataFrame(index = index, columns = columns)
    
    modelbase = model['base']
    
    trainednamer = lambda i: modelbase + '_%d' %i
    controlnamer = lambda i: modelbase + '_%dr' %i
    modelnames = [namer(i) for i in np.arange(1,6) for namer in (trainednamer, controlnamer)]
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    controlnames = [controlnamer(i) for i in np.arange(1,6)]
    
    from scipy.stats import ttest_rel
    for ilayer in layers:
        for tcname in enumerate(tcnames):
            trainedscores = []
            controlscores = []
            for name in modelnames:
                if name in trainednames:
                    trainedscores.append(df.loc[(name, ilayer, tcname), 'q90'])
                else:
                    controlscores.append(df.loc[(name, ilayer, tcname), 'q90'])
            pt_df.loc[(ilayer, tcname), 'pv'] = ttest_rel(trainedscores, controlscores)[1]
    
    analysisfolder = runinfo.sharedanalysisfolder(model, 'kindiffs')
    pt_df.to_csv(os.path.join(analysisfolder, model['base'] + '_pairedt_df.csv'))
    
def pairedt_comp(model, runinfo):
    nlayers = model['nlayers'] + 1
    modelbase = model['base']
    
    trials = list(np.arange(1,6))
    
    testtypes=['two-sided', 'less', 'greater']
    
    colnames = pd.MultiIndex.from_product((
            list(range(nlayers)),
            testtypes,
            ['p-value', 'sl']),
        names=('layer', 'testtype', 'sig_measure'))
    modelnames = [modelbase + '_%d' %i for i in np.arange(1,6)]
    
    index = pd.MultiIndex.from_product((
                modelnames,
                tcnames),
                names = ('model', 'tc'))
    
    df = pd.DataFrame(index=index, columns=colnames)
    
    for trial, mname in enumerate(modelnames):
        
        trainedmodel = model.copy()
        trainedmodel['name'] = mname
        controlmodel = model.copy()
        controlmodel['name'] = mname + 'r'
        
        trainedexpf={
              'vel': runinfo.resultsfolder(trainedmodel, 'vel'),
              'acc': runinfo.resultsfolder(trainedmodel, 'acc'),
              'labels': runinfo.resultsfolder(trainedmodel, 'labels')
        }
        
        controlexpf={
              'vel': runinfo.resultsfolder(controlmodel, 'vel'),
              'acc': runinfo.resultsfolder(controlmodel, 'acc'),
              'labels': runinfo.resultsfolder(controlmodel, 'labels')
        }
        
        for ilayer in np.arange(0,nlayers):
            
            dvevals = np.load(os.path.join(trainedexpf['vel'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'vel', mmod, runinfo.planestring())))
            accevals = np.load(os.path.join(trainedexpf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std', runinfo.planestring())))
            labevals = np.load(os.path.join(trainedexpf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
            
            trainedlayerevals = []
            trainedlayerevals.append(dvevals[...,1,1]) #dir
            trainedlayerevals.append(dvevals[...,2,1]) #vel
            trainedlayerevals.append(dvevals[...,3,1]) #dir + vel
            trainedlayerevals.append(accevals[...,2,1]) #acc
            trainedlayerevals.append(labevals[...,0]) #labels    

            dvevals = np.load(os.path.join(controlexpf['vel'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'vel', mmod, runinfo.planestring())))
            accevals = np.load(os.path.join(controlexpf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std', runinfo.planestring())))
            labevals = np.load(os.path.join(controlexpf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
            
            controllayerevals = []
            controllayerevals.append(dvevals[...,1,1]) #dir
            controllayerevals.append(dvevals[...,2,1]) #vel
            controllayerevals.append(dvevals[...,3,1]) #dir + vel
            controllayerevals.append(accevals[...,2,1]) #acc
            controllayerevals.append(labevals[...,0]) #labels    
            
            for itc, tc in enumerate(tcnames):
                #for testtype in testtypes:
                testtype='two-sided'
                from scipy.stats import ttest_rel
                pv = ttest_rel(trainedlayerevals[itc].flatten(), controllayerevals[itc].flatten())[1]
                df.loc[(trainedmodel['name'], tc), (ilayer, testtype, 'p-value')] = pv
                df.loc[(trainedmodel['name'], tc), (ilayer, testtype, 'sl')] = pv_to_sl_code(pv)
                
                df.loc[(trainedmodel['name'], tc), (ilayer, testtypes[1], 'p-value')] = trainedlayerevals[itc].mean()
                df.loc[(trainedmodel['name'], tc), (ilayer, testtypes[1], 'sl')] = controllayerevals[itc].mean()
    
    
    analysisfolder = runinfo.sharedanalysisfolder(model, 'pairedt')  
    os.makedirs(analysisfolder, exist_ok=True)
    df.to_csv(os.path.join(analysisfolder, 'pairedt.csv'))
    
    return df

def plot_pd_deviation(layers, tmdevs, cmdevs, trainedmodel):
    
    from scipy.stats import ttest_rel
    
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = fig.add_subplot(111)
        
    for i in range(len(tmdevs)):
        plt.plot(range(len(layers)), tmdevs[i], color=trainedmodel['color'], marker = 'D', alpha = 0.15, label='ind trained')
        plt.plot(range(len(layers)), cmdevs[i], color='grey', marker = 'D', alpha = 0.15, label='ind control')
    
    #solution to calculate conf. interval of means from https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html
    #t_corr = t.ppf(0.975, 4)
    
    tmsmean = np.nanmean(tmdevs, axis=0)    
    cmsmean = np.nanmean(cmdevs, axis=0)
    
    n_tms = np.sum(~np.isnan(tmdevs), axis=0)
    n_cms = np.sum(~np.isnan(cmdevs), axis=0)
    
    print(n_tms)
    print(n_cms)
    
    trained_t_corr = np.array([t.ppf(0.975, n - 1) for n in n_tms])
    control_t_corr = np.array([t.ppf(0.975, n - 1) for n in n_cms])
    
    errs_tmsmean = np.nanstd(tmdevs, axis=0) / np.sqrt(n_tms) * trained_t_corr
    errs_cmsmean = np.nanstd(cmdevs, axis=0) / np.sqrt(n_cms) * control_t_corr
    
    print(trained_t_corr)
    print(control_t_corr)
    
    plt.plot(range(len(layers)), tmsmean, color=trainedmodel['color'], marker = 'D')
    plt.plot(range(len(layers)), cmsmean, color='grey', marker = 'D')
    
    t_corr = t.ppf(0.975, 4)
    
    plt.errorbar(layers, tmsmean, yerr=errs_tmsmean, marker='D', color=trainedmodel['color'], capsize=3.0, label='mean of trained')
    plt.errorbar(layers, cmsmean, yerr=errs_cmsmean, marker = 'D', color='grey', capsize=3.0, label='mean of controls')

    plt.xticks(list(range(len(layers))))
    plt.xlim((-0.3, len(layers)-0.7))
    plt.xlabel('Layer')
    plt.ylabel('Total Absolute Deviation from Uniformity')
        
    ax = plt.gca()
    format_axis(ax)
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    
    plt.legend(handles[[0,1,10,11]], ['ind trained', 'ind control', \
                'mean of trained', 'mean of controls'])
    
    plt.tight_layout()
    
    return fig
    
def plot_pd_deviation_delta(layers, tmdevs, cmdevs, trainedmodel):
    
    from scipy.stats import ttest_rel
    
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = fig.add_subplot(111)
    
    delta = cmdevs - tmdevs
        
    for i in range(len(tmdevs)):
        plt.plot(layers, delta[i], color=trainedmodel['color'], marker = 'D', alpha = 0.15, label='ind trained')

    t_corr = t.ppf(0.975, 4)
    
    tmsmean = np.nanmean(delta, axis=0)
    errs_tmsmean = np.nanstd(delta, axis=0) / np.sqrt(5) * t_corr
    
    plt.plot(layers, tmsmean, color=trainedmodel['color'], marker = 'D')
    
    t_corr = t.ppf(0.975, 4)
    
    plt.errorbar(layers, tmsmean, yerr=errs_tmsmean, marker='D', color=trainedmodel['color'], capsize=3.0, label='mean of trained')

    plt.xticks(list(range(len(layers))))
    plt.xlim((-0.3, len(layers)-0.7))
    plt.xlabel('Layer')
    plt.ylabel('Delta Total Absolute Deviation')
    
    ax = plt.gca()
    format_axis(ax)
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    
    plt.legend(handles[[0,5]], ['ind' , \
                'mean'])
    
    plt.tight_layout()
    
    return fig
    
def pd_deviation(model, runinfo):
    nlayers = model['nlayers'] + 1
    
    layers = ['Sp.'] + ['L%d' %i for i in np.arange(1,nlayers)]

    modelbase = model['base']
    trainednamer = lambda i: modelbase + '_%d' %i
    controlnamer = lambda i: modelbase + '_%dr' %i
    modelnames = [namer(i) for i in np.arange(1,6) for namer in (trainednamer, controlnamer)]
    
    df = pd.DataFrame(index=modelnames, columns=layers)
    
    trainedmodeldevs = np.empty((5, nlayers))
    controlmodeldevs = np.empty((5, nlayers))
    
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    controlnames = [controlnamer(i) for i in np.arange(1,6)]
    
    ##initialize to nan
    trainedmodeldevs[:] = np.nan
    controlmodeldevs[:] = np.nan
    
    imodel = 0
    for mname in modelnames:
        model = model.copy()
        model['name'] = mname
        pds = get_pds_sp(model, runinfo, r2threshold = 0.2)
        
        for ilayer, layer in enumerate(layers):
            layerpds = pds[ilayer]
            if (layerpds != []):
                layermask = ~np.isnan(layerpds)
                layerpds = layerpds[layermask]
                
                ### normalize by number of directionally tuned neurons
                ndirtuned = len(layerpds)
            
                hist, binedges = np.histogram(layerpds, bins=18, range=(-np.pi, np.pi))
                
                histmean = np.nanmean(hist)
                layerdev = np.abs(hist - histmean).sum()

                ###normalize
                if ndirtuned > 0:
                    layerdev = layerdev / ndirtuned
                else:
                    layerdev = np.nan
                                    
                df.loc[mname, layer] = layerdev
                
                if mname[-1] != 'r':
                    trainedmodeldevs[imodel, ilayer] = layerdev
                else:
                    controlmodeldevs[imodel, ilayer] = layerdev     
        
        if mname[-1] == 'r':
            imodel += 1

    analysisfolder = runinfo.sharedanalysisfolder(model, 'pd_deviation')  
    os.makedirs(analysisfolder, exist_ok=True)
    df.to_csv(os.path.join(analysisfolder, 'pd_deviation_normalized.csv'))    
    
    fig = plot_pd_deviation(layers, trainedmodeldevs, controlmodeldevs, model)
    fig.savefig(os.path.join(analysisfolder, 'pd_dev_plot_normalized.pdf'))
    plt.close('all')
    
    fig = plot_pd_deviation_delta(range(nlayers), trainedmodeldevs, controlmodeldevs, model)
    fig.savefig(os.path.join(analysisfolder, 'pd_dev_delta_plot_normalized.pdf'))
    plt.close('all')
    
    
    ##statistical tests
    trained_df = df.loc[trainednames, :]
    control_df = df.loc[controlnames, :]
    n_comparisons = len(layers) - 1
    
    statsig_df = pd.DataFrame(index=['t stat', 'p-value', 'Bonferroni', 'n'], columns=layers)
    for layer in layers:
        traineddevs = trained_df[layer].values.astype(float)
        controldevs = control_df[layer].values.astype(float)
        
        trained_nanmask = ~np.isnan(traineddevs)
        control_nanmask = ~np.isnan(controldevs)
        
        nanmask = trained_nanmask * control_nanmask
        
        print(nanmask)
        
        t_stats = ttest_rel(traineddevs[nanmask], controldevs[nanmask])
        if layer == 7:
            assert t_stats, print('L7 failed', t_stats, traineddevs, controldevs)
        statsig_df.loc['t stat', layer] = t_stats[0]
        statsig_df.loc['p-value', layer] = t_stats[1]
        statsig_df.loc['Bonferroni', layer] = t_stats[1]*n_comparisons
        statsig_df.loc['n', layer] = sum(nanmask)
    statsig_df.to_csv(os.path.join(analysisfolder, 'statsig_df.csv'))
    
# %% PLOTS

def colorselector(cmapname, tcf, ct = 0.4):
    tcnames = ['dir', 'vel', 'dirvel', 'acc', 'labels']
    nmods = len(tcnames)
    tci = tcnames.index(tcf)
    
    cmap = matplotlib.cm.get_cmap(cmapname)
    cidx = tci*(1-ct)/(nmods-1)
    return cmap(cidx)

def colorselector_ee(cmapname, tcf, ct = 0.4):
    tcnames = ['ee', 'eepolar']
    nmods = len(tcnames)
    tci = tcnames.index(tcf)
    
    cmap = matplotlib.cm.get_cmap(cmapname)
    cidx = tci*(1-ct)/(nmods-1)
    return cmap(cidx)

def plotcomp(tcfdf, tcf, model):
    fig = plt.figure(figsize=(14,6), dpi=200)   
    
    trainednamer = lambda i: model['base'] + '_%d' %i
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    
    controlnamer = lambda i: model['base'] + '_%dr' %i
    controlnames = [controlnamer(i) for i in np.arange(1,6)]
    
    trainedcolor = colorselector(model['cmap'], tcf)
    controlcolor = colorselector('Greys_r', tcf)
    colors = [trainedcolor, controlcolor]
    
    x = range(model['nlayers'] + 1)
    
    for (names, color) in zip([trainednames, controlnames], colors):
        
        medians = tcfdf.loc[names, 'median']
        q90s = tcfdf.loc[names, 'q90']
        
        plt.plot(x, [medians.xs(i, level='layer')[1:].mean() for i in x], color=color, linestyle='-.', marker='D')
        plt.plot(x, [q90s.xs(i, level='layer')[1:].mean() for i in x ], color=color, marker='D')
        
        alpha = 0.15
        
        for name in names:
            
            plt.plot(x, medians.xs(name, level='model'), color=color, alpha=alpha, linestyle='-.', marker='D')
            plt.plot(x, q90s.xs(name, level='model'), color=color, alpha=alpha, marker='D')
        
        plt.title('%s tuning curve accuracies comparison for model and controls' %(tcf))
        plt.ylabel('r2 score')
        plt.xticks(x, ['spindles'] + ['layer %d' %i for i in np.arange(1,model['nlayers']+1)])
        plt.ylim((-0.1,1))
        
    ax = plt.gca()
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    plt.legend(handles[[0,1,2,3,12,13,14,15]], ['mean of trained medians', 'mean of trained q90s', 'mean of control medians', \
                'mean of control q90s', 'ind trained median', 'ind trained q90',
                'ind control median', 'ind control q90',])

    return fig

def plotcomp_dir_accs(tcfdf, tcf, model):

    fig = plt.figure(figsize=(12,5.5), dpi=300)   
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_axisbelow(True)
    
    trainednamer = lambda i: model['base'] + '_%d' %i
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    
    controlnamer = lambda i: model['base'] + '_%dr' %i
    controlnames = [controlnamer(i) for i in np.arange(1,6)]
    
    nlayers = model['nlayers']
    
    x = range(nlayers + 1)

    #solution to calculate conf. interval of means from https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html
    t_corr = t.ppf(0.975, 4)
    
    print(tcfdf.head())
    
    traineddirs = [tcfdf.loc[(trainednames, i, 'dir')].mean() for i in np.arange(nlayers+1)]
    controldirs = [tcfdf.loc[(controlnames, i, 'dir')].mean() for i in np.arange(nlayers+1)]
    errs_traineddirs = [tcfdf.loc[(trainednames, i, 'dir')].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controldirs = [tcfdf.loc[(controlnames, i, 'dir')].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    trainedlabels = [tcfdf.loc[(trainednames, i, 'acc')].mean() for i in np.arange(nlayers+1)]
    controllabels = [tcfdf.loc[(controlnames, i, 'acc')].mean() for i in np.arange(nlayers+1)]
    errs_trainedlabels = [tcfdf.loc[(trainednames, i, 'acc')].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controllabels = [tcfdf.loc[(controlnames, i, 'acc')].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    print(traineddirs)
    print(errs_traineddirs)
    
    plt.errorbar(x, traineddirs, yerr=errs_traineddirs, color=colorselector(model['cmap'], 'dir'), marker='D', capsize=3.0)
    plt.errorbar(x, controldirs, yerr=errs_controldirs, color=colorselector('Greys_r', 'dir'), marker='D', capsize=3.0)
    plt.errorbar(x, trainedlabels, yerr=errs_trainedlabels, color=colorselector(model['cmap'], 'acc'), linestyle='-.',marker='D', capsize=3.0)
    plt.errorbar(x, controllabels, yerr=errs_controllabels, color=colorselector('Greys_r', 'acc'), linestyle='-.', marker='D', capsize=3.0)
    plt.ylabel('r2 score')
    plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
               horizontalalignment = 'right')
    plt.ylim((-0.1,1))
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    plt.legend(['Dir Trained', 'Dir Controls', \
                'Acc Trained', 'Acc Controls'])
    
    ax = format_axis(ax)
    
    plt.tight_layout()

    return fig

def plotcomp_ees(tcfdf, model):

    fig = plt.figure(figsize=(12,5.5), dpi=300)   
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_axisbelow(True)
    
    trainednamer = lambda i: model['base'] + '_%d' %i
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    
    controlnamer = lambda i: model['base'] + '_%dr' %i
    controlnames = [controlnamer(i) for i in np.arange(1,6)]
    
    nlayers = model['nlayers']
    
    x = range(nlayers + 1)
        
    #solution to calculate conf. interval of means from https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html
    t_corr = t.ppf(0.975, 4)
    
    print(tcfdf.head())
    traineddirs = [np.nanmean(tcfdf.loc[(trainednames, i, 'ee')]) for i in np.arange(nlayers+1)]
    controldirs = [np.nanmean(tcfdf.loc[(controlnames, i, 'ee')]) for i in np.arange(nlayers+1)]
    errs_traineddirs = [np.nanstd(tcfdf.loc[(trainednames, i, 'ee')])/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controldirs = [np.nanstd(tcfdf.loc[(controlnames, i, 'ee')])/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    trainedlabels = [np.nanmean(tcfdf.loc[(trainednames, i, 'eepolar')]) for i in np.arange(nlayers+1)]
    controllabels = [np.nanmean(tcfdf.loc[(controlnames, i, 'eepolar')]) for i in np.arange(nlayers+1)]
    errs_trainedlabels = [np.nanstd(tcfdf.loc[(trainednames, i, 'eepolar')])/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controllabels = [np.nanstd(tcfdf.loc[(controlnames, i, 'eepolar')])/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    print(traineddirs)
    print(errs_traineddirs)
    
    plt.errorbar(x, traineddirs, yerr=errs_traineddirs, color=colorselector_ee(model['cmap'], 'ee'), marker='D', capsize=3.0)
    plt.errorbar(x, controldirs, yerr=errs_controldirs, color=colorselector_ee('Greys_r', 'ee'), marker='D', capsize=3.0)
    plt.errorbar(x, trainedlabels, yerr=errs_trainedlabels, color=colorselector_ee(model['cmap'], 'eepolar'), linestyle='-.',marker='D', capsize=3.0)
    plt.errorbar(x, controllabels, yerr=errs_controllabels, color=colorselector_ee('Greys_r', 'eepolar'), linestyle='-.', marker='D', capsize=3.0)
        
    plt.ylabel('r2 score')
    plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
               horizontalalignment = 'right')
    plt.ylim((-0.1,1))
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    plt.legend(['Cart Trained', 'Cart Controls', \
                'Polar Trained', 'Polar Controls'])
    
    ax = format_axis(ax)
    
    plt.tight_layout()

    return fig

def tcctrlcompplots(df, model, runinfo):
    folder = runinfo.sharedanalysisfolder(model,  'kindiffs_plots')
    os.makedirs(folder, exist_ok=True)
    tcf=None
    tcdf = df.loc[(slice(None), slice(None), ['dir', 'acc']), 'q90']#.reset_index(level=2, drop=True)
    
    fig = plotcomp_dir_accs(tcdf, tcf, model)
    fig.savefig(os.path.join(folder, 'tcctrlcomp_dir_accs_horlabels_shortlegend.pdf'))
    
    eedf = df.loc[(slice(None), slice(None), ['ee', 'eepolar']), 'q90']
    
    ee_fig = plotcomp_ees(eedf, model)
    ee_fig.savefig(os.path.join(folder, 'tcctrlcomp_ees.pdf'))
    
    plt.close('all')
        
# %% COMPARE PREF DIR DIFFS AND GENERALIZATION
        
def pv_to_string(pv):
    sl = 0
    if(pv < 0.1):
        sl = 1
        if(pv < 0.05):
            sl = 2
            if(pv < 0.01):
                sl = 3       
    return sl

def get_pds(model, runinfo, r2threshold = None):
    """ Return heights including 'all' """
    
    modelname = model['name']
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    base = model['base']
    
    fset = 'vel'
    mmod = 'std'
    
    ##SAVE PDs & R2s
    pds = []
    for ilayer in range(nlayers):
        pds.append([])
        for ior3, orientation3 in enumerate(orientations):
            pds[ilayer].append([])
            
            runinfo['orientation'] = orientation3
            runinfo['height'] = 'all'
            
            resultsfolder = runinfo.resultsfolder(model)
            expf = '%s%s/' %(resultsfolder, 'vel')
            
            testevals = np.load('%sl%d_%s_mets_%s_%s_test.npy' %(expf, ilayer, fset, mmod, runinfo.planestring()))        
            
            if r2threshold is not None:
                testevals = testevals[testevals[...,1,1] > r2threshold] #exclude those that fall below 0.2 threshold
            dirtuning = testevals[...,1,3:5].reshape((-1,2))            
            prefdirs = np.apply_along_axis(angle_xaxis, 1, dirtuning)
            
            """Create numpy for pref dirs with structure:
                Ax 1: Height Index
                Ax 2: Neurons
            """
            
            pds[ilayer][ior3] = np.zeros((len(uniqueheights[ior3]) + 1, len(prefdirs)))
            pds[ilayer][ior3][0] = prefdirs
            
            for iht, ht in enumerate(uniqueheights[ior3]):
                runinfo['height'] = ht

                testevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(runinfo.resultsfolder(model, 'vel'), ilayer, fset, mmod, runinfo.planestring()))
                dirtuning = testevals[...,1,3:5].reshape((-1,2))  
                prefdirs = np.apply_along_axis(angle_xaxis, 1, dirtuning)
                pds[ilayer][ior3][iht+1] = prefdirs
                
    return pds

def getlpdcors(pds):
    """args: preferred directions for all planes in a single layer"""
     ## CALCULATE LAYER-WISE CORRELATIONS
    nzs = len(uniquezs)
    nxs = len(uniquexs)
    zpos0 = uniquezs.index(0)
    xpos0 = int(len(uniquexs)/2)
    x0 = uniquexs[xpos0]
    z0 = 0
    
    # HOR VS HOR
    horvshor = np.zeros((nzs,))
    for iz in range(nzs):
        pd1 = pds[0][zpos0]
        pd2 = pds[0][iz]
        mask = np.logical_and(np.invert(np.isnan(pd1)), np.invert(np.isnan(pd2)))
        horvshor[iz] = np.corrcoef(pd1[mask], pd2[mask])[0,1]
            
    # VERT VS VERT
    vertvsvert = np.zeros((nxs,))
    for ix in range(nxs):
        pd1 = pds[1][xpos0]
        pd2 = pds[1][ix]
        mask = np.logical_and(np.invert(np.isnan(pd1)), np.invert(np.isnan(pd2)))
        vertvsvert[ix] = np.corrcoef(pd1, pd2)[0,1]
    
    #  HOR VS VERT (compare one horizontal to all verticals)
    horvsvert = np.zeros((nxs,))
    for ix in range(nxs):
        pd1 = pds[0][zpos0]
        pd2 = pds[1][ix]
        mask = np.logical_and(np.invert(np.isnan(pd1)), np.invert(np.isnan(pd2)))
        horvsvert[ix] = np.corrcoef(pd1[mask], pd2[mask])[0,1]
    
    # VERT VS HOR (one vertical vs. all horizontals)
    vertvshor = np.zeros((nzs,))
    for iz in range(nzs):
        pd1 = pds[1][xpos0]
        pd2 = pds[0][iz]
        mask = np.logical_and(np.invert(np.isnan(pd1)), np.invert(np.isnan(pd2)))
        vertvshor[iz] = np.corrcoef(pd1[mask], pd2[mask])[0,1]
        
    return [horvshor, vertvsvert, horvsvert, vertvshor], pd1.size

def get_r2s(model, runinfo, r2threshold  = 0.2):
    modelname = model['name']    
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    base = model['base']
    
    fset = 'vel'
    mmod = 'std'
        
    ##SAVE R2s
    r2s = []
    for ilayer in range(nlayers):
        r2s.append([])
        for ior2, orientation2 in enumerate(orientations):
            r2s[ilayer].append([])
            
            runinfo['orientation'] = orientation2
            runinfo['height'] = uniqueheights[ior2][0]
            
            resultsfolder = runinfo.resultsfolder(model)
            expf = '%s%s/' %(resultsfolder, 'vel')
            
            testevals = np.load('%sl%d_%s_mets_%s_%s_test.npy' %(expf, ilayer, fset, mmod, runinfo.planestring()))        

            r2 = testevals[...,1,1].reshape((-1))            
            
            """Create numpy for pref dirs with structure:
                Ax 1: Height Index
                Ax 2: Neurons
            """
            r2s[ilayer][ior2] = np.zeros((len(uniqueheights[ior2]), len(r2)))
            
            for iht, ht in enumerate(uniqueheights[ior2]):
                runinfo['height'] = ht

                testevals = np.load('%s/l%d_%s_mets_%s_%s_test.npy' %(runinfo.resultsfolder(model, 'vel'), ilayer, fset, mmod, runinfo.planestring()))
                r2 = testevals[...,1,1].reshape((-1))            
                r2s[ilayer][ior2][iht] = r2
                
    return r2s
        
def invarcomp(model, runinfo):    
    
    nlayers = model['nlayers'] + 1
    modelbase = model['base']
    modelnames = [modelbase + '_%d' %i for i in np.arange(1,6)]
    
    corrtype = ['trained', 'control', 'z-test']    
    
    colnames=list(zip(['hor']*len(uniquezs)*3 + ['vert']*len(uniquexs)*3,
                 np.concatenate((np.repeat(np.array(uniquezs),3), np.repeat(np.array(uniquexs),3))),
                 corrtype*(len(uniquezs)+len(uniquexs))))
    columns = pd.MultiIndex.from_tuples(colnames,
                names=('orientation', 'height', 'type'))
    
    index = pd.MultiIndex.from_product((
                modelnames,
                list(range(nlayers)),
                compors,
                corrtypes),
                names = ('model', 'layer', 'compor', 'corrtypes'))
    
    df = pd.DataFrame(index=index, columns=colnames)
    
    nlayers = model['nlayers'] + 1 #add 1 for spindles
    
    ##SAVE PDs & R2s
    for im, mname in enumerate(modelnames):
        
        trainedmodel = model.copy()
        trainedmodel['name'] = mname
        trainedpds = get_pds(trainedmodel, runinfo)
        
        controlmodel = model.copy()
        controlmodel['name'] = mname + 'r'
        controlpds = get_pds(controlmodel, runinfo)
            
        zpos0 = uniquezs.index(0)
        xpos0 = int(len(uniquexs)/2)
        
        for ilayer in np.arange(nlayers):
            trained = trainedpds[ilayer]
            #print(trained.shape)
            trainedcorrs, ntrained = getlpdcors(trained)
            #print(trainedcorrs[0].shape)
            ctrl = controlpds[ilayer]
            ctrlcorrs, ncorr = getlpdcors(ctrl)
            assert ntrained == ncorr, 'Size of train and control not equal!!!!'
            
            ior = 0
            orientation = 'hor'
            for iht, ht in enumerate(uniqueheights[ior]):
                hhtc = trainedcorrs[0][iht]
                vhtc = trainedcorrs[3][iht]
                hhcc = ctrlcorrs[0][iht]
                vhcc = ctrlcorrs[3][iht]
                df.loc[(mname, ilayer, compors[0], 'pd corr'), (orientation, ht, 'trained')] = hhtc
                df.loc[(mname, ilayer, compors[0], 'pd corr'), (orientation, ht, 'control')] = hhcc
                df.loc[(mname, ilayer, compors[0], 'pd corr'), (orientation, ht, 'z-test')] = independent_corr(hhtc, hhcc, ntrained)[1]
                
                df.loc[(mname, ilayer, compors[3], 'pd corr'), (orientation, ht, 'trained')] = vhtc
                df.loc[(mname, ilayer, compors[3], 'pd corr'), (orientation, ht, 'control')] = vhcc
                df.loc[(mname, ilayer, compors[3], 'pd corr'), (orientation, ht, 'z-test')] = independent_corr(vhtc, vhcc, ntrained)[1]
            
            ior = 1
            orientation = 'vert'
            for iht, ht in enumerate(uniqueheights[ior]):
                vvtc = trainedcorrs[1][iht]
                hvtc = trainedcorrs[2][iht]
                vvcc = ctrlcorrs[1][iht]
                hvcc = ctrlcorrs[2][iht]
                df.loc[(mname, ilayer, compors[1], 'pd corr'), (orientation, ht, 'trained')] = vvtc
                df.loc[(mname, ilayer, compors[1], 'pd corr'), (orientation, ht, 'control')] = vvcc
                df.loc[(mname, ilayer, compors[1], 'pd corr'), (orientation, ht, 'z-test')] = independent_corr(vvtc, vvcc, ntrained)[1]
                
                df.loc[(mname, ilayer, compors[2], 'pd corr'), (orientation, ht, 'trained')] = hvtc
                df.loc[(mname, ilayer, compors[2], 'pd corr'), (orientation, ht, 'control')] = hvcc
                df.loc[(mname, ilayer, compors[2], 'pd corr'), (orientation, ht, 'z-test')] = independent_corr(hvtc, hvcc, ntrained)[1]
            
        trainedr2s = get_r2s(trainedmodel, runinfo)
        
        controlr2s = get_r2s(controlmodel, runinfo)
            
        zpos0 = uniquezs.index(0)
        xpos0 = int(len(uniquexs)/2)
        
        for ilayer in np.arange(nlayers):
            trained = trainedr2s[ilayer]
            #print(trained.shape)
            trainedcorrs, ntrained = getlpdcors(trained)
            ctrl = controlr2s[ilayer]
            ctrlcorrs, ncorr = getlpdcors(ctrl)
            assert ntrained == ncorr, 'Size of train and control not equal!!!!'
            
            ior = 0
            orientation = 'hor'
            for iht, ht in enumerate(uniqueheights[ior]):
                hhtc = trainedcorrs[0][iht]
                vhtc = trainedcorrs[3][iht]
                hhcc = ctrlcorrs[0][iht]
                vhcc = ctrlcorrs[3][iht]
                df.loc[(mname, ilayer, compors[0], 'r2 corr'), (orientation, ht, 'trained')] = hhtc
                df.loc[(mname, ilayer, compors[0], 'r2 corr'), (orientation, ht, 'control')] = hhcc
                if(ht != zpos0):
                    df.loc[(mname, ilayer, compors[0], 'r2 corr'), (orientation, ht, 'z-test')] = independent_corr(hhtc, hhcc, ntrained)[1]
                
                df.loc[(mname, ilayer, compors[3], 'r2 corr'), (orientation, ht, 'trained')] = vhtc
                df.loc[(mname, ilayer, compors[3], 'r2 corr'), (orientation, ht, 'control')] = vhcc
                if(ht != zpos0):
                    df.loc[(mname, ilayer, compors[3], 'r2 corr'), (orientation, ht, 'z-test')] = independent_corr(vhtc, vhcc, ntrained)[1]
            
            ior = 1
            orientation = 'vert'
            for iht, ht in enumerate(uniqueheights[ior]):
                vvtc = trainedcorrs[1][iht]
                hvtc = trainedcorrs[2][iht]
                vvcc = ctrlcorrs[1][iht]
                hvcc = ctrlcorrs[2][iht]
                df.loc[(mname, ilayer, compors[1], 'r2 corr'), (orientation, ht, 'trained')] = vvtc
                df.loc[(mname, ilayer, compors[1], 'r2 corr'), (orientation, ht, 'control')] = vvcc
                if(ht != xpos0):
                    df.loc[(mname, ilayer, compors[1], 'r2 corr'), (orientation, ht, 'z-test')] = independent_corr(vvtc, vvcc, ntrained)[1]
                
                df.loc[(mname, ilayer, compors[2], 'r2 corr'), (orientation, ht, 'trained')] = hvtc
                df.loc[(mname, ilayer, compors[2], 'r2 corr'), (orientation, ht, 'control')] = hvcc
                if(ht != xpos0):
                    df.loc[(mname, ilayer, compors[2], 'r2 corr'), (orientation, ht, 'z-test')] = independent_corr(hvtc, hvcc, ntrained)[1]
            
    analysisfolder = runinfo.sharedanalysisfolder(model, 'invars', sp= False)
    os.makedirs(analysisfolder, exist_ok=True)
    df.to_csv(os.path.join(analysisfolder, model['base'] + '_invars_df.csv'))
        
    return df    

def plot_ind_neuron_invars_comp(layers, tmdevmean, cmdevmean, trainedmodel):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    
    plt.plot(layers, np.sum(np.abs(tmdevmean), axis=1), color=trainedmodel['color'], marker = 'D')
    plt.plot(layers, np.sum(np.abs(cmdevmean), axis=1), color='grey', marker = 'D')
    
    plt.legend(['trained', 'control'])
    
    plt.xlabel('Layer')
    plt.ylabel('Total Absolute Deviation')
    
    plt.title('Total Absolute Deviation for All Tuned Neurons')
    
    return fig

def plot_inic_am(layers, alltmdevmeans, allcmdevmeans, trainedmodel):
    
    #swtiched everything to nanmean...
    
    columns = ['tmsmean', 'cmsmean', 'tmsstd', 'cmsstd', 'stddiff', 't stat', 'p value', 'Bonferroni', 'N for t']
    index = layers
    df = pd.DataFrame(index=index, columns=columns)
    nlayers = trainedmodel['nlayers'] + 1
    
    from scipy.stats import ttest_rel
    
    fig = plt.figure(figsize=(8,6),dpi=300)
        
    for i in range(len(alltmdevmeans)):
        assert np.nanmax(alltmdevmeans[i]) < np.pi, 'too large tmdevmean, model %d, value %f, pos %s ' %(i, np.nanmax(alltmdevmeans[i]), np.argmax(alltmdevmeans[i]))
        plt.plot(layers, np.nanmean(np.abs(alltmdevmeans[i]), axis=1), color=trainedmodel['color'], marker = 'D', alpha = 0.15, label='ind trained')
        plt.plot(layers, np.nanmean(np.abs(allcmdevmeans[i]), axis=1), color='grey', marker = 'D', alpha = 0.15, label='ind control')
    
    #solution to calculate conf. interval of means from https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html
    t_corr = t.ppf(0.975, 4)
    
    n_tms = np.sum(~np.isnan(np.nanmean(np.abs(alltmdevmeans), axis=2)), axis=0)
    n_cms = np.sum(~np.isnan(np.nanmean(np.abs(allcmdevmeans), axis=2)), axis=0)
    
    print(n_tms)
    print(n_cms)
    
    trained_t_corr = np.array([t.ppf(0.975, n - 1) for n in n_tms])
    control_t_corr = np.array([t.ppf(0.975, n - 1) for n in n_cms])
    
    print(trained_t_corr)
    print(control_t_corr)
        
    tmsmean = np.nanmean(np.nanmean(np.abs(alltmdevmeans), axis=2), axis=0)
    cmsmean = np.nanmean(np.nanmean(np.abs(allcmdevmeans), axis=2), axis=0)
    errs_tmsmean = np.nanstd(np.nanmean(np.abs(alltmdevmeans), axis=2), axis=0) / np.sqrt(n_tms) * trained_t_corr
    errs_cmsmean = np.nanstd(np.nanmean(np.abs(allcmdevmeans), axis=2), axis=0) / np.sqrt(n_cms) * control_t_corr
    
    print(alltmdevmeans.shape)
    print(np.sum(np.abs(alltmdevmeans), axis=2).shape)
    
    df['tmsmean'] = tmsmean
    df['cmsmean'] = cmsmean
    df['tmsstd'] = np.nanstd(np.nanmean(np.abs(alltmdevmeans), axis=2), axis=0)
    df['cmsstd'] = np.nanstd(np.nanmean(np.abs(allcmdevmeans), axis=2), axis=0)
    df['stddiff'] = np.nanstd(np.nanmean(np.abs(alltmdevmeans), axis=2) - np.sum(np.abs(allcmdevmeans), axis=2), axis=0)
    
    #statistical significance
    tms = np.nanmean(np.abs(alltmdevmeans), axis=2) #axis 0: models, axis 1: layers
    cms = np.nanmean(np.abs(allcmdevmeans), axis=2)
    print('trained', tms)
    print('control', cms)
    n_comparisons = len(layers) - 1
    for layer in layers:
        print('L%d' %layer)
        tml = tms[:, layer]
        cml = cms[:, layer]
        
        tml_nanmask = ~np.isnan(tml)
        cml_nanmask = ~np.isnan(cml)
        
        nanmask = tml_nanmask * cml_nanmask
        
        tml = tml[nanmask]
        cml = cml[nanmask]
        
        print('tml', tml)
        print('cml', cml)
        t_stats = ttest_rel(tml, cml)
        df.loc[layer, 't stat'] = t_stats[0]
        df.loc[layer, 'p value'] = t_stats[1]
        df.loc[layer, 'Bonferroni'] = t_stats[1]*n_comparisons
        df.loc[layer, 'N for t'] = sum(nanmask)
        
        if layer == 7:
            assert t_stats, print('L7 failed!', t_stats, tml, cml)
        
        print(t_stats)
        time.sleep(0.5)
    
    plt.plot(layers, tmsmean, color=trainedmodel['color'], marker = 'D', label='mean of trained')
    plt.plot(layers, cmsmean, color='grey', marker = 'D', label='mean of controls')

    plt.xticks(list(range(len(layers))), ['Sp.'] + ['L%d' %(i+1) for i in range(len(layers))])
    plt.xlim((-0.3,len(layers)-0.7))
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Deviation')
    
    ax = plt.gca()
    format_axis(ax)
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    
    plt.legend(handles[[0,1,10,11]], ['ind trained', 'ind control', \
                'mean of trained', 'mean of controls'])
    
    plt.tight_layout()
    
    figeb = plt.figure(figsize=(8,6),dpi=300)
    
    plt.errorbar(layers, tmsmean, yerr=errs_tmsmean, marker='D', color=trainedmodel['color'], capsize=3.0)
    plt.errorbar(layers, cmsmean, yerr=errs_cmsmean, marker = 'D', color='grey', capsize=3.0)
    
    ax = plt.gca()
    format_axis(ax)
    
    plt.xticks(list(range(len(layers))), ['Sp.'] + ['L%d' %(i+1) for i in range(len(layers))])
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Deviation')
    plt.xlim((-0.3, len(layers)-0.7))
    
    plt.legend(['trained', 'control'])
    
    figboth = plt.figure(figsize=(8,6), dpi=300)
        
    for i in range(len(alltmdevmeans)):
        plt.plot(layers, np.nanmean(np.abs(alltmdevmeans[i]), axis=1), color=trainedmodel['color'], marker = 'D', alpha = 0.15, label='ind trained')
        plt.plot(layers, np.nanmean(np.abs(allcmdevmeans[i]), axis=1), color='grey', marker = 'D', alpha = 0.15, label='ind control')
    
    #solution to calculate conf. interval of means from https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html
    t_corr = t.ppf(0.975, 4)
    
    plt.errorbar(layers, tmsmean, yerr=errs_tmsmean, marker='D', color=trainedmodel['color'], capsize=3.0, label='mean of trained')
    plt.errorbar(layers, cmsmean, yerr=errs_cmsmean, marker = 'D', color='grey', capsize=3.0, label='mean of controls')

    plt.xticks(list(range(len(layers))), ['Sp.'] + ['L%d' %(i+1) for i in range(len(layers))])
    plt.xlim((-0.3, len(layers)-0.7))
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Deviation')
    
    ax = plt.gca()
    format_axis(ax)
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    
    plt.legend(handles[[0,1,10,11]], ['ind trained', 'ind control', \
                'mean of trained', 'mean of controls'])
    
    return fig, figeb, figboth, df


def ind_neuron_invars_comp(model, runinfo):    
    
    nlayers = model['nlayers'] + 1
    modelbase = model['base']
    modelnames = [modelbase + '_%d' %i for i in np.arange(1,6)]
    
    
    analysisfolder = runinfo.sharedanalysisfolder(model, 'ind_neuron_invars_comp', False)
    os.makedirs(analysisfolder, exist_ok = True)
     
    ##SAVE PDs & R2s
    
    alltraineddevsim_hor = []
    alltraineddevsim_vert = []
    allcontroldevsim_hor = []
    allcontroldevsim_vert = []
    
    for im, mname in enumerate(modelnames):
        
        trainedmodel = model.copy()
        trainedmodel['name'] = mname
        
        controlmodel = model.copy()
        controlmodel['name'] = mname + 'r'
            
        zpos0 = uniquezs.index(0)
        xpos0 = int(len(uniquexs)/2)
        
                    
        ior = 0
        orientation = 'hor'
        tmf = runinfo.generalizationfolder(trainedmodel, 'ind_neuron_invar_collapsed_beautified')
        tmdevmean_hor = pd.read_csv(os.path.join(tmf, 'ind_neuron_invar_hor_deviations_02.csv'), index_col = 0).values #mean absolute deviation over neurons saved here
        tmdevmean_vert = pd.read_csv(os.path.join(tmf, 'ind_neuron_invar_vert_deviations_02.csv'), index_col = 0).values
        
        tmdevmean_hor[:,zpos0] = np.nan
        tmdevmean_vert[:,zpos0] = np.nan
        
        alltraineddevsim_hor.append(tmdevmean_hor)
        alltraineddevsim_vert.append(tmdevmean_vert)
        
        cmf = runinfo.generalizationfolder(controlmodel, 'ind_neuron_invar_collapsed_beautified')
        cmdevmean_hor = pd.read_csv(os.path.join(cmf, 'ind_neuron_invar_hor_deviations_02.csv'), index_col = 0).values
        cmdevmean_vert = pd.read_csv(os.path.join(cmf, 'ind_neuron_invar_vert_deviations_02.csv'), index_col = 0).values    
        
        cmdevmean_hor[:,zpos0] = np.nan
        cmdevmean_vert[:,zpos0] = np.nan
        
        allcontroldevsim_hor.append(cmdevmean_hor)
        allcontroldevsim_vert.append(cmdevmean_vert)
        
        fig_hor = plot_ind_neuron_invars_comp(list(range(nlayers)), tmdevmean_hor, cmdevmean_hor, trainedmodel)
        fig_vert = plot_ind_neuron_invars_comp(list(range(nlayers)), tmdevmean_vert, cmdevmean_vert, trainedmodel)
        
        fig_hor.savefig(os.path.join(analysisfolder, trainedmodel['name'] + '_ind_neuron_dev_plot_hor_02.png'))
        fig_vert.savefig(os.path.join(analysisfolder, trainedmodel['name'] + '_ind_neuron_dev_plot_vert_02.png'))
    
    fig_hor, figeb_hor, figboth_hor, df_hor = plot_inic_am(list(range(nlayers)), np.stack(alltraineddevsim_hor), np.stack(allcontroldevsim_hor), trainedmodel)
    fig_vert, figeb_vert, figboth_vert, df_vert = plot_inic_am(list(range(nlayers)), np.stack(alltraineddevsim_vert), np.stack(allcontroldevsim_vert), trainedmodel)
    
    fig_hor.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_hor_02.pdf'))
    figeb_hor.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_hor_eb_02.pdf'))
    fig_vert.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_vert_02.pdf'))
    figeb_vert.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_vert_eb_02.pdf'))
    figboth_hor.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_hor_both_02.pdf'))
    figboth_vert.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_vert_both_02.pdf'))
    
    df_hor.to_csv(os.path.join(analysisfolder, modelbase + '_deviations_mad_sig_hor.csv'))
    df_vert.to_csv(os.path.join(analysisfolder, modelbase + '_deviations_mad_sig_vert.csv'))
        
    print('plots saved')


# %% COMPARE ALL MODEL TYPES ACROSS FEATURES
    
def tab_comp_tcn_allmodels(tcn, allmodels, runinfo):
    
    modeltypes = [model['type'] for model in allmodels]
    trials = np.arange(1,6)
    
    layers = ['L%d' %i for i in np.arange(1,9)]
    
    index = pd.MultiIndex.from_product((
                    modeltypes,
                    trials),
                    names = ('modeltype', 'trial'))
        
    df = pd.DataFrame(index=index, columns=layers + ['total greater (alpha=0.05)', 'total lesser (alpha=0.05)', 'total'])
    
    for basemodel in allmodels:
        modelksfolder = runinfo.sharedanalysisfolder(basemodel, 'kindiffs')  
        modelks = pd.read_csv(os.path.join(modelksfolder, 'ks.csv'), index_col=[0,1], header=[0,1,2],)
        #print(modelks.columns)
        for i in trials:
            modelname = basemodel['base'] + '_%d' %i
            
            counterslless = 0
            counterslgreater = 0,
            for il in range(basemodel['nlayers']+1):
                layer = 'L%d' %il
                slless = modelks.loc[(modelname, tcn), (str(il), 'less', 'sl')]
                slgreater = modelks.loc[(modelname, tcn), (str(il), 'greater', 'sl')]
                
                slstring = ''
                for j in range(int(slless)):
                    slstring = slstring + '+'
                for j in range(int(slgreater)):
                    slstring = slstring + '-'
                df.loc[(basemodel['type'], i), layer] = slstring
        
                if(slless >= 2):
                    counterslless = counterslless + 1
                if(slgreater >= 2):
                    counterslgreater = counterslgreater + 1
                    
    allmodelfolder = runinfo.allmodelfolder(analysis = 'kindiffs')
    os.makedirs(allmodelfolder, exist_ok=True)
    df.to_csv(os.path.join(allmodelfolder, 'allmodels_%s_ks_slsums.csv' %tcn))
    
    return df    
    
def tab_comp_pds_allmodels(allmodels, runinfo):
    
    modeltypes = [model['type'] for model in allmodels]
    trials = np.arange(1,6)
    
    layers = ['L%d' %i for i in np.arange(1,9)]
    
    index = pd.MultiIndex.from_product((
                    modeltypes,
                    trials),
                    names = ('modeltype', 'trial'))
        
    df = pd.DataFrame(index=index, columns=layers)
    
    for basemodel in allmodels:
        modelksfolder = runinfo.sharedanalysisfolder(basemodel, 'pddiffs')  
        modelks = pd.DataFrame.from_csv(os.path.join(modelksfolder, 'pddiffs.csv'), index_col=[0,1], header=0)
       
        for i in trials:
            modelname = basemodel['base'] + '_%d' %i
            for il in range(basemodel['nlayers']+1):
                layer = 'L%d' %il
                pv = modelks.loc[(modelname, il), 'kuiper']
                sl = pv_to_sl_code(convert_str_to_float(pv))
                
                df.loc[(basemodel['type'], i), layer] = ''.join(['*']*int(sl))
                    
    allmodelfolder = runinfo.allmodelfolder(analysis = 'pddiffs')
    os.makedirs(allmodelfolder, exist_ok=True)
    df.to_csv(os.path.join(allmodelfolder, 'allmodels_pddiffs.csv'))
    
    return df

def tab_comp_invars_allmodels(corrtype, allmodels, runinfo):
    
    modeltypes = [model['type'] for model in allmodels]
    trials = np.arange(1,6)
    
    layers = ['L%d' %i for i in np.arange(1,9)]
    planeranges = ['lower', 'middle', 'upper']
    
    columns = pd.MultiIndex.from_product((
                    layers,
                    planeranges),
                    names = ('layers', 'planeranges'))
        
    planerangecutoffs = np.array([[-36, 21], #cutoffs are < center <=, lower <=, higher >
                                  [15, 45]]) #row 1: horizontal vs. horizontal
                                             #row 2: vertical vs. vertical
    
    index = pd.MultiIndex.from_product((
                    modeltypes,
                    trials),
                    names = ('modeltype', 'trial'))
        
    df = pd.DataFrame(index=index, columns=columns)
    
    for basemodel in allmodels:
        modelksfolder = runinfo.sharedanalysisfolder(basemodel, 'invars', sp=False)  
        modelks = pd.DataFrame.from_csv(os.path.join(modelksfolder, '%s_invars_df.csv' %basemodel['base'] ), index_col=[0,1,2,3], header=0, )
        #print(modelks.index)
        #print(modelks.columns)
        for i in trials:
            modelname = basemodel['base'] + '_%d' %i
            for il in range(basemodel['nlayers']+1):
                layer = 'L%d' %il
                for ior, orientation in enumerate(orientations):
                    counterheights = 0
                    counterslgreater = np.zeros((3,))
                    countersllesser = np.zeros((3,))
                    counterrange = 0
                    for iht, ht in enumerate(uniqueheights[ior]):
                        trained = modelks.loc[(modelname, il, compors[ior], corrtype), str((orientation, ht, 'trained'))]
                        control = modelks.loc[(modelname, il, compors[ior], corrtype), str((orientation, ht, 'control'))]
                        zscore = modelks.loc[(modelname, il, compors[ior], corrtype),str((orientation, ht, 'z-test'))]
                        
                        sl = pv_to_sl_code(zscore)

                        if(sl):
                            
                            if(trained > control):
                                for j in range(sl):
                                    counterslgreater[j] = counterslgreater[j] + 1
                            else:
                                for j in range(sl):
                                    counterslgreater[j] = counterslgreater[j] + 1
                        
                        counterheights = counterheights + 1
                        
                        if((iht == len(uniqueheights[ior]) - 1) or (counterrange < 2 and ht == planerangecutoffs[ior, counterrange])):
                            morethanhalfgreater = np.where(counterslgreater >= counterheights/2.0)[0]
                            morethanhalflesser = np.where(countersllesser >= counterheights/2.0)[0]
                            
                            assert ~((morethanhalfgreater.size > 0) and (morethanhalflesser.size > 0)), 'more than half can\'t both be greater and lesser!'
                            
                            if(morethanhalfgreater.size > 0):
                                sl = morethanhalfgreater[len(morethanhalfgreater) -1] + 1
                                df.loc[(basemodel['type'], i), (layer, planeranges[counterrange])] = ''.join(['+']*int(sl))
                            elif(morethanhalflesser.size > 0):
                                sl = morethanhalflesser[len(morethanhalflesser) -1] + 1
                                df.loc[(basemodel['type'], i), (layer, planeranges[counterrange])] = ''.join(['-']*int(sl))
                                
                            counterheights = 0
                            counterslgreater = np.zeros((3,))
                            countersllesser = np.zeros((3,))
                            counterrange = counterrange + 1
                            
    allmodelfolder = runinfo.allmodelfolder(analysis = 'invars', sp=False)
    os.makedirs(allmodelfolder, exist_ok=True)
    df.to_csv(os.path.join(allmodelfolder, 'allmodels_invars_%s.csv' %corrtype))
    
    return df 

# %% MAIN

def main(model, runinfo):
    """main for comparing features in a single plane for all models"""
    
    print('comparing kinetic differences for model %s ...' %model['base'])
    df = None
    
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'kindiffs'))):
    if(True):
        print('compiling dataframe for comparions...')
        df = compile_comparisons_df(model, runinfo)
        
    else:
        print('kinetic and label embeddings already analyzed')
        
    if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'pairedt'))):
    #if(True):
        
        print('running pairedt...')
        pairedt_comp(model, runinfo)
        print('ks analysis saved')
        
    else:
        print('kinetic and label embeddings already analyzed')
        
    if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'kindiffs_plots'))):
    #if(True):
        if df is None:
            analysisfolder = runinfo.sharedanalysisfolder(model, 'kindiffs')
            df = pd.read_csv(os.path.join(analysisfolder, model['base'] + '_comparisons_df.csv'),
                             header=0, index_col=[0,1,2], dtype={'layer': int, 'mean': float, 'median': float})
        print('creating kindiffs plots')
        tcctrlcompplots(df, model, runinfo)
    else:
        print('kindiffs plots already made')
        
    if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'pd_deviation'))):
    #if(True):
        print('computing deviation measure for PDs')
        pd_deviation(model, runinfo)
        print('df saved')
    else:
        print('pd deviation measure already saved')
        
def generalizations_comparisons_main(model, runinfo):
    """main for comparing features that describe all planes"""
    
    if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'invars', False))):
    #if(True):
        print('analyzing siginifance of differences in preferred direction and generalization for %s models...' %model['base'])
        invarcomp(model, runinfo)
        print('Dataframe saved')
    else:
        print('pd invars already analyzed')
    
    if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'ind_neuron_invars_comp', False))):
    #if(True):
        print('running individual neuron invars comparison...')
        ind_neuron_invars_comp(model, runinfo)
        print('saved plots')
    else:
        print('ind neuron invars comps already run')
        
    plt.close('all')
        
def compare_all_types_main(allmodels, runinfo):
    if(not os.path.exists(runinfo.allmodelfolder('kindiffs'))):
    #if(True):
        print('comparing all model types with their controls for kinematic tuning...')
        for tcn in tcnames:
            tab_comp_tcn_allmodels(tcn, allmodels, runinfo)
        print('completed')
    else:
        print('joint comparison all model types and controls for kinematic tuning already completed')
    
    if(not os.path.exists(runinfo.allmodelfolder('pddiffs'))):
    #if(True):
        print('comparing pds for all model types and controls...')
        tab_comp_pds_allmodels(allmodels, runinfo)
    else:
        print('joint comparison all model types and controls for PDs already completed')
        
    if(not os.path.exists(runinfo.allmodelfolder('pdvars'))):
        print('comparing variance in PDs for all model types and controls...')
        tab_comp_pdvars_allmodels(allmodels, runinfo)
    else:
        print('variance in PDs already compared for all model types and controls')
        
    if(not os.path.exists(runinfo.allmodelfolder('pddiffs_to_orig'))):
    #if(True):
        print('comparing all model types to original...')
        tab_comp_pddiffs_to_orig_allmodels(allmodels, runinfo)
        
    if(not os.path.exists(runinfo.allmodelfolder('pddiffs_rayleigh'))):
    #if(True):
        print('comparing all models\' uniformity of PD distribution...')
        tab_comp_pddiffs_rayleigh_allmodels(allmodels, runinfo)
        
def invars_all_types_main(allmodels, runinfo):
    if(not os.path.exists(runinfo.allmodelfolder('invars', sp=False))):
        for corrtype in corrtypes:
            print('comparing all model types with their controls for kinematic tuning...')
            tab_comp_invars_allmodels(corrtype, allmodels, runinfo)
    else:
        print('joint comparison all model types and controls for invariance already completed')
