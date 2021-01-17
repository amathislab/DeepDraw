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
import matplotlib.pyplot as plt
from rowwise_neuron_curves_controls import *
import os
from scipy.stats import t, ttest_rel
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
    """ Get pds in a single plane that are siginificantly tuned for direction, i.e. r2threshold < r2_score < 1
    
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
            testevals = testevals[(testevals[...,1,1] > r2threshold) & (testevals[...,1,1] != 1)]  #also check for r2 != 1
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
    ''' Formats a significance level int to string showing level by strings and 
    
    Arguments
    ---------
    slgreater : float, two-sided significance level or one-sided upper significance level
    sllesser : float, one-sided significance level or one-sided lower significance level
    
    Returns
    -------
    slstring : formatted string showing significance level via symbols
    '''
    
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
    ''' Compiles and saves a pandas dataframe that brings together various different metrics 
    for all model instantiations within a particular model type
    
    Arguments
    ---------
    model : dict
    runinfo : RunInfo (extension of dict)
    '''
    
    nlayers = model['nlayers'] + 1
    
    colnames = ['mean', 'median', 'std', 'max', 'min', 'q90', 'q10']#, 'ee', 'eepolar']
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
              'ee': runinfo.resultsfolder(model_to_analyse, 'ee'),
              'eepolar': runinfo.resultsfolder(model_to_analyse, 'eepolar')
        }
        
        for ilayer in np.arange(0,nlayers):
            
            dvevals = np.load(os.path.join(expf['vel'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'vel', mmod, runinfo.planestring())))
            accevals = np.load(os.path.join(expf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std', runinfo.planestring())))
            labevals = np.load(os.path.join(expf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
            eeevals = np.load(os.path.join(expf['ee'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'ee', 'std', runinfo.planestring())))
            eepolarevals = np.load(os.path.join(expf['eepolar'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'eepolar', 'std', runinfo.planestring())))

            layerevals = []
            layerevals.append(dvevals[...,1,1]) #dir
            layerevals.append(dvevals[...,2,1]) #vel
            layerevals.append(dvevals[...,3,1]) #dir + vel
            layerevals.append(accevals[...,2,1]) #acc
            layerevals.append(labevals[...,0]) #labels
            layerevals.append(eeevals[...,0,1]) #ee
            layerevals.append(eepolarevals[...,3,1]) #eepolar
            
            for j, tcname in enumerate(tcnames):

                #exclude r2 == 1 scores
                layerevals[j] = layerevals[j][layerevals[j] != 1]

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
    
    assert not eestats_df.empty, "Endeffector Dataframe empty!!!"
    eestats_df.to_csv(os.path.join(analysisfolder, model['base'] + '_eestats_df.csv'))
    
    return df

def pairedt_quantiles(df, model, runinfo):
    ''' Saves a dataframe with results from paired ttest comapring 90 percent quantiles of different kinematic tuning curves
    between trained and control models of a particular model type
    
    Arguments
    ---------
    df : pd.DataFrame matching output of compile_comparisons
    model : dict
    runinfo : RunInfo (extension of dict)
    
    
    '''
        
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
    ''' Saves a dataframe with results from paired ttest comapring individual test scores of different 
    kinematic tuning curves between trained and control models of a particular model type
    
    Arguments
    ---------
    df : pd.DataFrame matching output of compile_comparisons
    model : dict
    runinfo : RunInfo (extension of dict)
    
    '''
    
    nlayers = model['nlayers'] + 1
    modelbase = model['base']
        
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

            for i in range(len(trainedlayerevals)):
                trainedlayerevals[i][trainedlayerevals[i] == 1] = np.nan

            dvevals = np.load(os.path.join(controlexpf['vel'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'vel', mmod, runinfo.planestring())))
            accevals = np.load(os.path.join(controlexpf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std', runinfo.planestring())))
            labevals = np.load(os.path.join(controlexpf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
            
            controllayerevals = []
            controllayerevals.append(dvevals[...,1,1]) #dir
            controllayerevals.append(dvevals[...,2,1]) #vel
            controllayerevals.append(dvevals[...,3,1]) #dir + vel
            controllayerevals.append(accevals[...,2,1]) #acc
            controllayerevals.append(labevals[...,0]) #labels 

            for i in range(len(controllayerevals)):
                controllayerevals[i][controllayerevals[i] == 1] = np.nan
            
            #print(tcnames)
            for itc, tc in enumerate(tcnames[:5]):
                #for testtype in testtypes:
                testtype='two-sided'
                #print(itc, len(trainedlayerevals))
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
    ''' Plot TAD from uniformity of preferred directions (sum over bins of histogram)
    in comparison between trained and control model instantations for a particular model type
    
    Arguments
    ---------
    layers : list of ints, different layer numbers
    tmdevs : np.array [nr model instantiations, nr of bins], deviation scores for trained model instantiations
    cmdevs : np.array [nr model instantiations, nr of bins], deviation scores for control model instantiations
    trainedmodel : dict, information on the trained model
    
    Returns
    -------
    fig : plt.figure, TAD plot
    
    '''
        
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
    
def pd_deviation(model, runinfo):
    ''' Compare deviation in preferred directions for a particular model type between trained and controls.
    Calls plotting functions and saves statistical significance to CSV file
    
    Arguments
    ---------
    model : dict, information about model type
    runinfo : RunInfo (extension of dict)
    
    
    '''
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
    ''' Helper function to select color for kinematic tuning curve types and labels
    
    Arguments
    ---------
    cmapname : str, the name of the color map that is being used 
    tcf : str, name of tuning feature that is being plotted
    ct : int, opt, specifies range of colorbar that is used
    
    Returns
    -------
    color values
    '''
    
    tcnames = ['dir', 'vel', 'dirvel', 'acc', 'labels']
    nmods = len(tcnames)
    tci = tcnames.index(tcf)
    
    cmap = matplotlib.cm.get_cmap(cmapname)
    cidx = tci*(1-ct)/(nmods-1)
    return cmap(cidx)

def colorselector_ee(cmapname, tcf, ct = 0.4):
    ''' Helper function to select color for positional tuning curve types and labels
    
    Arguments
    ---------
    cmapname : str, the name of the color map that is being used 
    tcf : str, name of tuning feature that is being plotted
    ct : int, opt, specifies range of colorbar that is used
    
    Returns
    -------
    color values
    '''
    tcnames = ['ee', 'eepolar']
    nmods = len(tcnames)
    tci = tcnames.index(tcf)
    
    cmap = matplotlib.cm.get_cmap(cmapname)
    cidx = tci*(1-ct)/(nmods-1)
    return cmap(cidx)

def plotcomp_dir_accs(tcfdf, tcf, model):
    ''' Plot comparisons between 90% quantiles for trained and control models for direction and acceleration 
    on a single plot
    
    Arguments
    ---------
    tcfdf : pd.DataFrame, index and columns matching output of compile_comparisons
    tcf : str, name of tuning feature
    model : dict
    
    Returns
    -------
    fig : plt.figure
    '''
    
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
    ''' Plot comparisons between 90% quantiles for trained and control models for endeffector position
    
    Arguments
    ---------
    tcfdf : pd.DataFrame, index and columns matching output of compile_comparisons
    tcf : str, name of tuning feature
    model : dict
    
    Returns
    -------
    fig : plt.figure
    '''
    
    print('Endeffectors model comparisons plot')
    
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
    print(tcfdf.shape)
    #print([tcfdf.loc[(trainednames, i, 'ee')])
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
    ''' Saves plots comparing 90% quantiles for the model for various tuning feature types
    
    Arguments
    ---------
    df : pd.DataFrame matching output of compile_comparisons
    model : dict
    runinfo : RunInfo (extension of dict)
    
    '''
    
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

def plot_inic_am(layers, alltmdevmeans, allcmdevmeans, trainedmodel):
    '''Plot MAD from PD in central plane over all planes
    Comparison between trained and control models
    
    Arguments
    ---------
    layers : list of ints, specifying layer indices
    alltmdevmeans : np.array [nr model instantiations, nr layers, nr planes], MAD scores for trained model
    allcmdevmeans : np.array [nr model instantiations, nr layers, nr planes], MAD scores for control model
    trainedmodel : dict
        
    Returns
    -------
    figboth : plt.figure
    df : pd.DataFrame, statistics accompanying plot
    '''
    
    columns = ['tmsmean', 'cmsmean', 'tmsstd', 'cmsstd', 'stddiff', 't stat', 'p value', 'Bonferroni', 'N for t']
    index = layers
    df = pd.DataFrame(index=index, columns=columns)
    
    n_tms = np.sum(~np.isnan(np.nanmean(np.abs(alltmdevmeans), axis=2)), axis=0)
    n_cms = np.sum(~np.isnan(np.nanmean(np.abs(allcmdevmeans), axis=2)), axis=0)
    
    trained_t_corr = np.array([t.ppf(0.975, n - 1) for n in n_tms])
    control_t_corr = np.array([t.ppf(0.975, n - 1) for n in n_cms])
    
    tmsmean = np.nanmean(np.nanmean(np.abs(alltmdevmeans), axis=2), axis=0)
    cmsmean = np.nanmean(np.nanmean(np.abs(allcmdevmeans), axis=2), axis=0)
    errs_tmsmean = np.nanstd(np.nanmean(np.abs(alltmdevmeans), axis=2), axis=0) / np.sqrt(n_tms) * trained_t_corr
    errs_cmsmean = np.nanstd(np.nanmean(np.abs(allcmdevmeans), axis=2), axis=0) / np.sqrt(n_cms) * control_t_corr
    
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
        
        #print(t_stats)
        #time.sleep(0.5)
    
    figboth = plt.figure(figsize=(8,6), dpi=300)
        
    for i in range(len(alltmdevmeans)):
        plt.plot(layers, np.nanmean(np.abs(alltmdevmeans[i]), axis=1), color=trainedmodel['color'], marker = 'D', alpha = 0.15, label='ind trained')
        plt.plot(layers, np.nanmean(np.abs(allcmdevmeans[i]), axis=1), color='grey', marker = 'D', alpha = 0.15, label='ind control')
        
    plt.errorbar(layers, tmsmean, yerr=errs_tmsmean, marker='D', color=trainedmodel['color'], capsize=3.0, label='mean of trained')
    plt.errorbar(layers, cmsmean, yerr=errs_cmsmean, marker = 'D', color='grey', capsize=3.0, label='mean of controls')

    plt.xticks(list(range(len(layers))), ['Sp.'] + ['L%d' %(i+1) for i in range(len(layers) - 1)])
    plt.xlim((-0.3, len(layers)-0.7))
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Deviation')
    
    ax = plt.gca()
    format_axis(ax)
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    
    plt.legend(handles[[0,1,10,11]], ['ind trained', 'ind control', \
                'mean of trained', 'mean of controls'])
    
    return figboth, df


def ind_neuron_invars_comp(model, runinfo):    
    ''' Compare individual neuron invariance for a given model type
    Save plots and statistics
    
    Arguments
    ---------
    model : dict
    runinfo : RunInfo (extension of dict)
    '''
    
    nlayers = model['nlayers'] + 1
    modelbase = model['base']
    modelnames = [modelbase + '_%d' %i for i in np.arange(1,6)]
    
    
    analysisfolder = runinfo.sharedanalysisfolder(model, 'ind_neuron_invars_comp', False)
     
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
    
    os.makedirs(analysisfolder, exist_ok = True)

    figboth_hor, df_hor = plot_inic_am(list(range(nlayers)), np.stack(alltraineddevsim_hor), np.stack(allcontroldevsim_hor), trainedmodel)
    figboth_vert, df_vert = plot_inic_am(list(range(nlayers)), np.stack(alltraineddevsim_vert), np.stack(allcontroldevsim_vert), trainedmodel)
    
    figboth_hor.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_hor_both_02.pdf'))
    figboth_vert.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_vert_both_02.pdf'))
    
    df_hor.to_csv(os.path.join(analysisfolder, modelbase + '_deviations_mad_sig_hor.csv'))
    df_vert.to_csv(os.path.join(analysisfolder, modelbase + '_deviations_mad_sig_vert.csv'))
        
    print('plots saved')

# %% MAIN

def main(model, runinfo):
    """main for comparing features in a single plane for all models"""
    
    print('comparing kinetic differences for model %s ...' %model['base'])
    df = None
    
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'kindiffs'))):
    if(True):
    #if(runinfo.default_run):
        print('compiling dataframe for comparions...')
        df = compile_comparisons_df(model, runinfo)
        
    else:
        print('kinetic and label embeddings already analyzed')
        
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'pairedt'))):
    if(True):
    #if(runinfo.default_run):    
        print('running pairedt...')
        pairedt_comp(model, runinfo)
        print('ks analysis saved')
        
    else:
        print('kinetic and label embeddings already analyzed')
        
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'kindiffs_plots'))):
    if(True):
    #if(runinfo.default_run):
        if df is None:
            analysisfolder = runinfo.sharedanalysisfolder(model, 'kindiffs')
            df = pd.read_csv(os.path.join(analysisfolder, model['base'] + '_comparisons_df.csv'),
                             header=0, index_col=[0,1,2], dtype={'layer': int, 'mean': float, 'median': float})
        print('creating kindiffs plots')
        tcctrlcompplots(df, model, runinfo)
    else:
        print('kindiffs plots already made')

    if(runinfo.default_run):    
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'pd_deviation'))):
    #if(True):
        print('computing deviation measure for PDs')
        pd_deviation(model, runinfo)
        print('df saved')
    else:
        print('pd deviation measure already saved')
        
def generalizations_comparisons_main(model, runinfo):
    """main for comparing features that describe all planes"""
    
    if(runinfo.default_run):
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'ind_neuron_invars_comp', False))):
    #if(True):
        print('running individual neuron invars comparison...')
        ind_neuron_invars_comp(model, runinfo)
        print('saved plots')
    else:
        print('ind neuron invars comps already run')
        
    plt.close('all')

# %%
