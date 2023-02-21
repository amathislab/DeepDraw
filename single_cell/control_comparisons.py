#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:31:24 2019

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
#dec_tcnames = ['ee_x', 'ee_y', 'eepolar_r', 'eepolar_theta', 'vel', 'dir', 'acc_r', 'acc_theta', 'labels']
dec_tcnames = ['ee_x', 'ee_y', 'eepolar_r', 'eepolar_theta', 'vel', 'dir', 'acc_r', 'acc_theta']
uniquezs = list(np.array([-45., -42., -39., -36., -33., -30., -27., -24., -21., -18., -15.,
                     -12.,  -9.,  -6.,  -3.,   0.,   3.,   6.,   9.,  12.,  15.,  18.,
                     21.,  24.,  27.,  30.]).astype(int))
uniquexs = list(np.array([ 6.,  9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42.,
                     45., 48., 51., 54.]).astype(int))
uniqueheights = [uniquezs, uniquexs]
orientations = ['hor', 'vert']

tcnames_fancy = {
    'dir': 'Dir.',
    'vel': 'Speed',
    'dirvel': 'Speed',
    'acc': 'Acc.',
    'labels': 'Labels',
    'ee': 'Pos.',
    'ee_polar': 'Pos. Polar'
}

dec_tcnames_fancy = {
    'ee_x': 'Pos. X',
    'ee_y': 'Pos. Y',
    'eepolar_r': 'Polar Pos.',
    'eepolar_theta': 'Polar Pos. Theta',
    'ee_mean': 'Mean Pos.',
    'vel': 'Speed',
    'dir': 'Dir.',
    'acc_r': 'Acc. R',
    'acc_theta': 'Acc. Theta'
}

corrtypes = ['pd corr', 'r2 corr']

compors = ['hors vs. hors', 'verts vs. verts', 'hor vs. verts', 'vert vs. hors']

#alphas = [0, 0.001, 0.01, 0.1, 1.0, 5.0]
#alphas = [0, 0.001, 0.01, 0.1, 1.0, 5.0, 10, 100, 1000, 10000, 100000, 1000000]
alphas = [1]
#alphas = [0.01, 1, 100, 10000]

def format_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

    ## SET AXIS WIDTHS
    for axis in ['top','bottom','left','right']:
        #ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_linewidth(2.5) ## eLife

    # increase tick width
    #ax.tick_params(width=1.5)
    ax.tick_params(width=2.5) ## eLife
    
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f')) ## eLife

# %% rcParams and Settings

details_size_factor = 6/5

params = {
   'axes.labelsize': 18,
   'legend.fontsize': 12,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
    #'axes.labelsize': 22,
    #'axes.fontsize': 18,
    #'legend.fontsize': 16,
    #'xtick.labelsize': 20,
    #'ytick.labelsize': 20,
    'text.usetex': False,
    'figure.figsize': [8,8 ],
    'font.size': 18,
    'axes.titlepad': 18,
    'xtick.major.size': 4.5*details_size_factor,
    'xtick.minor.size': 3.5*details_size_factor,
    'xtick.major.width': 1*details_size_factor,
    'ytick.major.size': 4.5*details_size_factor,
    'ytick.major.width': 1*details_size_factor,
    'ytick.minor.size': 3.5*details_size_factor
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

    print("COMPILING COMPARISONS DF")
    
    nlayers = model['nlayers'] + 1
    
    colnames = ['mean', 'median', 'std', 'max', 'min', 'q90', 'q10', 'excluded n', 'total n']#, 'ee', 'eepolar']
    #colnames = ['mean', 'median', 'std', 'max', 'min', 'q90', 'q10']#, 'ee', 'eepolar']
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
                original_layerevals = layerevals[j]
                layerevals[j] = layerevals[j][layerevals[j] != 1]

                df.loc[(mname, ilayer, tcname), 'mean'] = layerevals[j].mean()
                df.loc[(mname, ilayer, tcname), 'median'] = np.median(layerevals[j])
                df.loc[(mname, ilayer, tcname), 'std'] = layerevals[j].mean()               
                df.loc[(mname, ilayer, tcname), 'max'] = layerevals[j].max()                
                df.loc[(mname, ilayer, tcname), 'min'] = layerevals[j].min()                
                df.loc[(mname, ilayer, tcname), 'q90'] = np.quantile(layerevals[j], 0.9)
                df.loc[(mname, ilayer, tcname), 'q10'] = np.quantile(layerevals[j], 0.10)
                df.loc[(mname, ilayer, tcname), 'excluded n'] = sum((original_layerevals.flatten() == 1) | (original_layerevals.flatten() < -0.1))
                df.loc[(mname, ilayer, tcname), 'total n'] = original_layerevals.size
                #print("ADDING EXCLUDED N")

                if ilayer == 0:
                    firsthalf = layerevals[j][:layerevals[j].shape[0]//2]
                    secondhalf = layerevals[j][layerevals[j].shape[0]//2:]
                    df.loc[(mname, 11, tcname), 'mean'] = firsthalf.mean()
                    df.loc[(mname, 11, tcname), 'median'] = np.median(firsthalf)
                    df.loc[(mname, 11, tcname), 'std'] = firsthalf.mean()               
                    df.loc[(mname, 11, tcname), 'max'] = firsthalf.max()                
                    df.loc[(mname, 11, tcname), 'min'] = firsthalf.min()                
                    df.loc[(mname, 11, tcname), 'q90'] = np.quantile(firsthalf, 0.9)
                    df.loc[(mname, 11, tcname), 'q10'] = np.quantile(firsthalf, 0.10)
                    
                    df.loc[(mname, 12, tcname), 'mean'] = secondhalf.mean()
                    df.loc[(mname, 12, tcname), 'median'] = np.median(secondhalf)
                    df.loc[(mname, 12, tcname), 'std'] = secondhalf.mean()               
                    df.loc[(mname, 12, tcname), 'max'] = secondhalf.max()                
                    df.loc[(mname, 12, tcname), 'min'] = secondhalf.min()                
                    df.loc[(mname, 12, tcname), 'q90'] = np.quantile(secondhalf, 0.9)
                    df.loc[(mname, 12, tcname), 'q10'] = np.quantile(secondhalf, 0.10)
                
                
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


def compile_comparisons_tr_reg_df(taskmodel, regressionmodel, runinfo):
    ''' Compiles and saves a pandas dataframe that brings together various different metrics 
    for all model instantiations within a particular model type
    
    Arguments
    ---------
    model : dict
    runinfo : RunInfo (extension of dict)
    '''
    
    nlayers = taskmodel['nlayers'] + 1
    
    colnames = ['mean', 'median', 'std', 'max', 'min', 'q90', 'q10', 'excluded n', 'total n']#, 'ee', 'eepolar']
    #modelnames = [model['name']] + [model['name'] + '_%d' %(i + 1) for i in range(5)]
    taskmodelbase = taskmodel['base']
    regressionmodelbase = regressionmodel['base']
    print("Regression base: ", regressionmodelbase)
    
    trainednamer = lambda i: taskmodelbase + '_%d' %i
    controlnamer = lambda i: regressionmodelbase + '_%d' %i
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
        print(mname)
        
        if im%2==0:
            model_to_analyse = taskmodel.copy()
        else:
            print('using regression model base')
            model_to_analyse = regressionmodel.copy()
        model_to_analyse['name'] = mname
        #resultsfolder = runinfo.resultsfolder(model_to_analyse)
        
        expf={
              'vel': runinfo.resultsfolder(model_to_analyse, 'vel'),
              'acc': runinfo.resultsfolder(model_to_analyse, 'acc'),
              'labels': runinfo.resultsfolder(model_to_analyse, 'labels'),
              'ee': runinfo.resultsfolder(model_to_analyse, 'ee'),
              'eepolar': runinfo.resultsfolder(model_to_analyse, 'eepolar')
        }

        print('expf vel: ', expf)
        
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

                if ilayer == 0:
                    print("saving special stats for spindle input layer")
                    print("shape of layerevals[j]", layerevals[j].shape)
                    firsthalf = layerevals[j][:,0]
                    secondhalf = layerevals[j][:,1]
                    df.loc[(mname, 11, tcname), 'mean'] = firsthalf.mean()
                    df.loc[(mname, 11, tcname), 'median'] = np.median(firsthalf)
                    df.loc[(mname, 11, tcname), 'std'] = firsthalf.std()               
                    df.loc[(mname, 11, tcname), 'max'] = firsthalf.max()                
                    df.loc[(mname, 11, tcname), 'min'] = firsthalf.min()                
                    df.loc[(mname, 11, tcname), 'q90'] = np.quantile(firsthalf, 0.9)
                    df.loc[(mname, 11, tcname), 'q10'] = np.quantile(firsthalf, 0.10)
                    
                    df.loc[(mname, 12, tcname), 'mean'] = secondhalf.mean()
                    df.loc[(mname, 12, tcname), 'median'] = np.median(secondhalf)
                    df.loc[(mname, 12, tcname), 'std'] = secondhalf.std()               
                    df.loc[(mname, 12, tcname), 'max'] = secondhalf.max()                
                    df.loc[(mname, 12, tcname), 'min'] = secondhalf.min()                
                    df.loc[(mname, 12, tcname), 'q90'] = np.quantile(secondhalf, 0.9)
                    df.loc[(mname, 12, tcname), 'q10'] = np.quantile(secondhalf, 0.10)

                #exclude r2 == 1 scores
                levs_kept = layerevals[j][(layerevals[j] != 1) & (layerevals[j] > -0.1)]

                df.loc[(mname, ilayer, tcname), 'mean'] = levs_kept.mean()
                df.loc[(mname, ilayer, tcname), 'median'] = np.median(levs_kept)
                df.loc[(mname, ilayer, tcname), 'std'] = levs_kept.std()               
                df.loc[(mname, ilayer, tcname), 'max'] = levs_kept.max()                
                df.loc[(mname, ilayer, tcname), 'min'] = levs_kept.min()                
                df.loc[(mname, ilayer, tcname), 'q90'] = np.quantile(levs_kept, 0.9)
                df.loc[(mname, ilayer, tcname), 'q10'] = np.quantile(levs_kept, 0.10)
                df.loc[(mname, ilayer, tcname), 'excluded n'] = sum((layerevals[j].flatten() == 1) | (layerevals[j].flatten() < -0.1))
                df.loc[(mname, ilayer, tcname), 'total n'] = layerevals[j].size
                
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
                    
    analysisfolder = runinfo.sharedanalysisfolder(taskmodel, 'kindiffs_tr_reg')
    os.makedirs(analysisfolder, exist_ok=True)
    df.to_csv(os.path.join(analysisfolder, taskmodel['base'] + '_comparisons_reg_tr_df.csv'))
    
    labelstats_df.to_csv(os.path.join(analysisfolder, taskmodel['base'] + '_labelstats_reg_tr_df.csv'))
    
    assert not eestats_df.empty, "Endeffector Dataframe empty!!!"
    eestats_df.to_csv(os.path.join(analysisfolder, taskmodel['base'] + '_eestats_reg_tr_df.csv'))
    
    return df

def compile_decoding_comparisons_df(model, runinfo, alpha=None):
    ''' Compiles and saves a pandas dataframe that brings together various different metrics 
    for all model instantiations within a particular model type
    
    Arguments
    ---------
    model : dict
    runinfo : RunInfo (extension of dict)
    '''
    
    nlayers = model['nlayers'] + 1
    
    colnames = ['r2', 'RMSE', 'PCC/dist']
    #modelnames = [model['name']] + [model['name'] + '_%d' %(i + 1) for i in range(5)]
    modelbase = model['base']
    
    trainednamer = lambda i: modelbase + '_%d' %i
    controlnamer = lambda i: modelbase + '_%dr' %i
    modelnames = [namer(i) for i in np.arange(1,6) for namer in (trainednamer, controlnamer)]
    
    index = pd.MultiIndex.from_product((
                modelnames,
                list(range(nlayers)),
                dec_tcnames),
                names = ('model', 'layer', 'tc'))
    df = pd.DataFrame(index=index, columns=colnames)
    
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    
    for im, mname in enumerate(modelnames):
        
        model_to_analyse = model.copy()
        model_to_analyse['name'] = mname
        #resultsfolder = runinfo.resultsfolder(model_to_analyse)
                
        expf={
              'decoding_ee': runinfo.resultsfolder(model_to_analyse, 'decoding_ee'),
              'decoding_eepolar': runinfo.resultsfolder(model_to_analyse, 'decoding_eepolar'),
              'decoding_vel': runinfo.resultsfolder(model_to_analyse, 'decoding_vel'),
              'decoding_acc': runinfo.resultsfolder(model_to_analyse, 'decoding_acc'),
              'decoding_labels': runinfo.resultsfolder(model_to_analyse, 'decoding_labels'),
        }
        
        for ilayer in np.arange(0,nlayers):
            
            #SWITCHED FOR NORMALIZATION
            #decoding_ee_evals = np.load(os.path.join(expf['decoding_ee'], 'l%d_%s_mets_%s_%s_normalized_test.npy' %( ilayer, 'ee', 'decoding', runinfo.planestring())))
            #decoding_eepolar_evals = np.load(os.path.join(expf['decoding_eepolar'], 'l%d_%s_mets_%s_%s_normalized_test.npy' %(ilayer, 'eepolar', 'decoding', runinfo.planestring())))
            #decoding_vel_evals = np.load(os.path.join(expf['decoding_vel'], 'l%d_%s_mets_%s_%s_normalized_test.npy' %(ilayer, 'vel', 'decoding', runinfo.planestring())))
            #decoding_acc_evals = np.load(os.path.join(expf['decoding_acc'], 'l%d_%s_mets_%s_%s_normalized_test.npy' %(ilayer, 'acc', 'decoding', runinfo.planestring())))

            if alpha is None:
                decoding_ee_evals = np.load(os.path.join(expf['decoding_ee'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'ee', 'decoding', runinfo.planestring())))
                decoding_eepolar_evals = np.load(os.path.join(expf['decoding_eepolar'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'eepolar', 'decoding', runinfo.planestring())))
                decoding_vel_evals = np.load(os.path.join(expf['decoding_vel'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'vel', 'decoding', runinfo.planestring())))
                decoding_acc_evals = np.load(os.path.join(expf['decoding_acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'decoding', runinfo.planestring())))
                if 'labels' in dec_tcnames:
                    decoding_label_evals = np.load(os.path.join(expf['decoding_labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'decoding', runinfo.planestring())))
            else:
                decoding_ee_evals = np.load(os.path.join(expf['decoding_ee'], 'l%d_%s_mets_%s_%s_a%d_test.npy' %( ilayer, 'ee', 'decoding', runinfo.planestring(), int(alpha*1000))))
                decoding_eepolar_evals = np.load(os.path.join(expf['decoding_eepolar'], 'l%d_%s_mets_%s_%s_a%d_test.npy' %(ilayer, 'eepolar', 'decoding', runinfo.planestring(), int(alpha*1000))))
                decoding_vel_evals = np.load(os.path.join(expf['decoding_vel'], 'l%d_%s_mets_%s_%s_a%d_test.npy' %(ilayer, 'vel', 'decoding', runinfo.planestring(), int(alpha*1000))))
                decoding_acc_evals = np.load(os.path.join(expf['decoding_acc'], 'l%d_%s_mets_%s_%s_a%d_test.npy' %(ilayer, 'acc', 'decoding', runinfo.planestring(), int(alpha*1000))))
                if 'labels' in dec_tcnames:
                    decoding_label_evals = np.load(os.path.join(expf['decoding_labels'], 'l%d_%s_mets_%s_%s_a%d_test.npy' %(ilayer, 'labels', 'decoding', runinfo.planestring(), int(alpha*1000))))
        
            layerevals = []
            layerevals.append(decoding_ee_evals[0,1]) #ee_x
            layerevals.append(decoding_ee_evals[1,1]) #ee_y

            layerevals.append(decoding_eepolar_evals[0,1]) #eepolar_r
            layerevals.append(decoding_eepolar_evals[1,1]) #eepolar_theta

            layerevals.append(decoding_vel_evals[0,1]) #vel
            layerevals.append(decoding_vel_evals[1,1]) #dir

            layerevals.append(decoding_acc_evals[0,1]) #acc_r
            layerevals.append(decoding_acc_evals[1,1]) #acc_theta  

            if 'labels' in dec_tcnames:
                layerevals.append(decoding_label_evals[0,1]) #labels

            layerevals_RMSE = []
            layerevals_RMSE.append(decoding_ee_evals[0,0]) #ee_x
            layerevals_RMSE.append(decoding_ee_evals[1,0]) #ee_y

            layerevals_RMSE.append(decoding_eepolar_evals[0,0]) #eepolar_r
            layerevals_RMSE.append(decoding_eepolar_evals[1,0]) #eepolar_theta

            layerevals_RMSE.append(decoding_vel_evals[0,0]) #vel
            layerevals_RMSE.append(decoding_vel_evals[1,0]) #dir

            layerevals_RMSE.append(decoding_acc_evals[0,0]) #acc_r
            layerevals_RMSE.append(decoding_acc_evals[1,0]) #acc_theta  
        
            if 'labels' in dec_tcnames:
                layerevals_RMSE.append(decoding_label_evals[0,0]) #labels

            for j, tcname in enumerate(dec_tcnames):

                df.loc[(mname, ilayer, tcname), 'r2'] = layerevals[j]
                df.loc[(mname, ilayer, tcname), 'RMSE'] = layerevals_RMSE[j]
                df.loc[(mname, ilayer, tcname), 'PCC/dist'] = layerevals_RMSE[j]
                
    analysisfolder = runinfo.sharedanalysisfolder(model, 'decoding_kindiffs')
    os.makedirs(analysisfolder, exist_ok=True)

    #SWITCHED FOR NORMALIZATION
    #df.to_csv(os.path.join(analysisfolder, model['base'] + '_decoding_comparisons_df_normalized.csv'))

    if alpha is None:
        savefile = os.path.join(analysisfolder, model['base'] + '_decoding_comparisons_df.csv')
    else:
        savefile = os.path.join(analysisfolder, model['base'] + '_decoding_comparisons_df_a%d.csv' %alpha)

    #df.to_csv()
    df.to_csv(savefile)

    return df

def compile_decoding_comparisons_tr_reg_df(taskmodel, regressionmodel, runinfo, alpha = None):
    ''' Compiles and saves a pandas dataframe that brings together various different metrics 
    for all model instantiations within a particular model type
    
    Arguments
    ---------
    model : dict
    runinfo : RunInfo (extension of dict)
    '''
    
    nlayers = taskmodel['nlayers'] + 1
    
    #colnames = ['mean', 'median', 'std', 'max', 'min', 'q90', 'q10']#, 'ee', 'eepolar']
    #modelnames = [model['name']] + [model['name'] + '_%d' %(i + 1) for i in range(5)]
    colnames = ['r2', 'RMSE', 'PCC/dist']
    taskmodelbase = taskmodel['base']
    regressionmodelbase = regressionmodel['base']
    
    trainednamer = lambda i: taskmodelbase + '_%d' %i
    controlnamer = lambda i: regressionmodelbase + '_%d' %i
    modelnames = [namer(i) for i in np.arange(1,6) for namer in (trainednamer, controlnamer)]

    index = pd.MultiIndex.from_product((
                modelnames,
                list(range(nlayers)),
                dec_tcnames + ['ee_mean']),
                names = ('model', 'layer', 'tc'))
    df = pd.DataFrame(index=index, columns=colnames)
    
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    
    for im, mname in enumerate(modelnames):
    
        if im%2==0:
            model_to_analyse = taskmodel.copy()
        else:
            print('using regression model base')
            model_to_analyse = regressionmodel.copy()
        model_to_analyse['name'] = mname
        #resultsfolder = runinfo.resultsfolder(model_to_analyse)
                
        expf={
              'decoding_ee': runinfo.resultsfolder(model_to_analyse, 'decoding_ee'),
              'decoding_eepolar': runinfo.resultsfolder(model_to_analyse, 'decoding_eepolar'),
              'decoding_vel': runinfo.resultsfolder(model_to_analyse, 'decoding_vel'),
              'decoding_acc': runinfo.resultsfolder(model_to_analyse, 'decoding_acc'),
              'decoding_labels': runinfo.resultsfolder(model_to_analyse, 'decoding_labels'),
        }
        
        for ilayer in np.arange(0,nlayers):
            
            #SWITCHED FOR NORMALIZATION
            #decoding_ee_evals = np.load(os.path.join(expf['decoding_ee'], 'l%d_%s_mets_%s_%s_normalized_test.npy' %( ilayer, 'ee', 'decoding', runinfo.planestring())))
            #decoding_eepolar_evals = np.load(os.path.join(expf['decoding_eepolar'], 'l%d_%s_mets_%s_%s_normalized_test.npy' %(ilayer, 'eepolar', 'decoding', runinfo.planestring())))
            #decoding_vel_evals = np.load(os.path.join(expf['decoding_vel'], 'l%d_%s_mets_%s_%s_normalized_test.npy' %(ilayer, 'vel', 'decoding', runinfo.planestring())))
            #decoding_acc_evals = np.load(os.path.join(expf['decoding_acc'], 'l%d_%s_mets_%s_%s_normalized_test.npy' %(ilayer, 'acc', 'decoding', runinfo.planestring())))

            '''
            decoding_ee_evals = np.load(os.path.join(expf['decoding_ee'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'ee', 'decoding', runinfo.planestring())))
            decoding_eepolar_evals = np.load(os.path.join(expf['decoding_eepolar'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'eepolar', 'decoding', runinfo.planestring())))
            decoding_vel_evals = np.load(os.path.join(expf['decoding_vel'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'vel', 'decoding', runinfo.planestring())))
            decoding_acc_evals = np.load(os.path.join(expf['decoding_acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'decoding', runinfo.planestring())))
            #decoding_label_evals = np.load(os.path.join(expf['decoding_labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'decoding', runinfo.planestring())))
            '''
            
            if alpha is None:
                decoding_ee_evals = np.load(os.path.join(expf['decoding_ee'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'ee', 'decoding', runinfo.planestring())))
                decoding_eepolar_evals = np.load(os.path.join(expf['decoding_eepolar'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'eepolar', 'decoding', runinfo.planestring())))
                decoding_vel_evals = np.load(os.path.join(expf['decoding_vel'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'vel', 'decoding', runinfo.planestring())))
                decoding_acc_evals = np.load(os.path.join(expf['decoding_acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'decoding', runinfo.planestring())))
            else:
                decoding_ee_evals = np.load(os.path.join(expf['decoding_ee'], 'l%d_%s_mets_%s_%s_a%d_test.npy' %( ilayer, 'ee', 'decoding', runinfo.planestring(), int(alpha*1000))))
                decoding_eepolar_evals = np.load(os.path.join(expf['decoding_eepolar'], 'l%d_%s_mets_%s_%s_a%d_test.npy' %(ilayer, 'eepolar', 'decoding', runinfo.planestring(), int(alpha*1000))))
                decoding_vel_evals = np.load(os.path.join(expf['decoding_vel'], 'l%d_%s_mets_%s_%s_a%d_test.npy' %(ilayer, 'vel', 'decoding', runinfo.planestring(), int(alpha*1000))))
                decoding_acc_evals = np.load(os.path.join(expf['decoding_acc'], 'l%d_%s_mets_%s_%s_a%d_test.npy' %(ilayer, 'acc', 'decoding', runinfo.planestring(), int(alpha*1000))))
        
            layerevals = []
            layerevals.append(decoding_ee_evals[0,1]) #ee_x
            layerevals.append(decoding_ee_evals[1,1]) #ee_y

            layerevals.append(decoding_eepolar_evals[0,1]) #eepolar_r
            layerevals.append(decoding_eepolar_evals[1,1]) #eepolar_theta

            layerevals.append(decoding_vel_evals[0,1]) #vel
            layerevals.append(decoding_vel_evals[1,1]) #dir

            layerevals.append(decoding_acc_evals[0,1]) #acc_r
            layerevals.append(decoding_acc_evals[1,1]) #acc_theta  

            #layerevals.append(decoding_label_evals[0,1]) #labels

            layerevals_RMSE = []
            layerevals_RMSE.append(decoding_ee_evals[0,0]) #ee_x
            layerevals_RMSE.append(decoding_ee_evals[1,0]) #ee_y

            layerevals_RMSE.append(decoding_eepolar_evals[0,0]) #eepolar_r
            layerevals_RMSE.append(decoding_eepolar_evals[1,0]) #eepolar_theta

            layerevals_RMSE.append(decoding_vel_evals[0,0]) #vel
            layerevals_RMSE.append(decoding_vel_evals[1,0]) #dir

            layerevals_RMSE.append(decoding_acc_evals[0,0]) #acc_r
            layerevals_RMSE.append(decoding_acc_evals[1,0]) #acc_theta  

            layerevals_PCC = []
            layerevals_PCC.append(decoding_ee_evals[0,2]) #ee_x
            layerevals_PCC.append(decoding_ee_evals[1,2]) #ee_y

            layerevals_PCC.append(decoding_eepolar_evals[0,2]) #eepolar_r
            layerevals_PCC.append(decoding_eepolar_evals[1,2]) #eepolar_theta

            layerevals_PCC.append(decoding_vel_evals[0,2]) #vel
            layerevals_PCC.append(decoding_vel_evals[1,2]) #dir

            layerevals_PCC.append(decoding_acc_evals[0,2]) #acc_r
            layerevals_PCC.append(decoding_acc_evals[1,2]) #acc_theta  
        
            #layerevals_PCC.append(decoding_labels_evals[0,0]) #labels

            for j, tcname in enumerate(dec_tcnames):

                df.loc[(mname, ilayer, tcname), 'r2'] = layerevals[j]
                df.loc[(mname, ilayer, tcname), 'RMSE'] = layerevals_RMSE[j]
                df.loc[(mname, ilayer, tcname), 'PCC/dist'] = layerevals_PCC[j]

            df.loc[(mname, ilayer, 'ee_mean'), 'r2'] = (layerevals[0]+layerevals[1])/2
            df.loc[(mname, ilayer, 'ee_mean'), 'RMSE'] = (layerevals_RMSE[0]+layerevals_RMSE[1])/2
            df.loc[(mname, ilayer, 'ee_mean'), 'PCC/dist'] = (layerevals_PCC[0]+layerevals_PCC[1])/2
                
    analysisfolder = runinfo.sharedanalysisfolder(taskmodel, 'decoding_kindiffs')
    os.makedirs(analysisfolder, exist_ok=True)

    #SWITCHED FOR NORMALIZATION
    #df.to_csv(os.path.join(analysisfolder, taskmodel['base'] + '_decoding_comparisons_df_normalized.csv'))

    if alpha is None:
        savefile = os.path.join(analysisfolder, taskmodel['base'] + '_decoding_comparisons_tr_reg_df.csv')
    else:
        savefile = os.path.join(analysisfolder, taskmodel['base'] + '_decoding_comparisons_tr_reg_df_a%d.csv' %int(alpha*1000))

    df.to_csv(savefile)
        
    return df

def pairedt_quantiles(df, model, runinfo, regressionmodel = None):
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
    #columns = ['pv', 'n']
    columns = ['t stat', 'p-value', 'Bonferroni', 'n']
    pt_df = pd.DataFrame(index = index, columns = columns)
    
    modelbase = model['base']
    
    trainednamer = lambda i: modelbase + '_%d' %i
    
    if regressionmodel is not None:
        regressionmodelbase = regressionmodel['base']
        #regressionmodelnames = [regressionmodelbase + '_%d' %i for i in np.arange(1,6)]
        controlnamer = lambda i: regressionmodelbase + '_%d' %i 
    else:
        controlnamer = lambda i: modelbase + '_%dr' %i 

    modelnames = [namer(i) for i in np.arange(1,6) for namer in (trainednamer, controlnamer)]
    trainednames = [trainednamer(i) for i in np.arange(1,6)]

    n_comparisons = nlayers
    
    from scipy.stats import ttest_rel
    for ilayer in layers:
        for tcname in tcnames:
            trainedscores = []
            controlscores = []
            for name in modelnames:
                if name in trainednames:
                    trainedscores.append(df.loc[(name, ilayer, tcname), 'q90'])
                else:
                    controlscores.append(df.loc[(name, ilayer, tcname), 'q90'])

            print('trainedscores', trainedscores)
            print('controlscores', controlscores)
            t_stats = ttest_rel(trainedscores, controlscores)
            print('tstats', t_stats)
            pt_df.loc[(ilayer, tcname), 't stat'] = t_stats[0]
            pt_df.loc[(ilayer, tcname), 'p-value'] = t_stats[1]
            pt_df.loc[(ilayer, tcname), 'Bonferroni'] = t_stats[1]*n_comparisons
            pt_df.loc[(ilayer, tcname), 'n'] = 5

    if regressionmodel is None:
        analysisfolder = runinfo.sharedanalysisfolder(model, 'kindiffs')
    else:
        analysisfolder = runinfo.sharedanalysisfolder(model, 'kindiffs_tr_reg')
    pt_df.to_csv(os.path.join(analysisfolder, model['base'] + '_pairedt_df.csv'))
    
def pairedt_comp(model, runinfo, regressionmodel = None):
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

    if regressionmodel is not None:
        regressionmodelbase = regressionmodel['base']
        regressionmodelnames = [regressionmodelbase + '_%d' %i for i in np.arange(1,6)]
    
    index = pd.MultiIndex.from_product((
                modelnames,
                tcnames),
                names = ('model', 'tc'))
    
    df = pd.DataFrame(index=index, columns=colnames)
    
    for trial, mname in enumerate(modelnames):
        
        trainedmodel = model.copy()
        trainedmodel['name'] = mname
        if regressionmodel is None:
            controlmodel = model.copy()
            controlmodel['name'] = mname + 'r'
        else:
            controlmodel = regressionmodel.copy()
            controlmodel['name'] = regressionmodelnames[trial] + 'r'
        
        trainedexpf={
              'vel': runinfo.resultsfolder(trainedmodel, 'vel'),
              'ee': runinfo.resultsfolder(trainedmodel, 'ee'),
              'acc': runinfo.resultsfolder(trainedmodel, 'acc'),
              'labels': runinfo.resultsfolder(trainedmodel, 'labels')
        }
        
        controlexpf={
              'vel': runinfo.resultsfolder(controlmodel, 'vel'),
              'ee': runinfo.resultsfolder(controlmodel, 'ee'),
              'acc': runinfo.resultsfolder(controlmodel, 'acc'),
              'labels': runinfo.resultsfolder(controlmodel, 'labels')
        }
        
        for ilayer in np.arange(0,nlayers):
            
            dvevals = np.load(os.path.join(trainedexpf['vel'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'vel', mmod, runinfo.planestring())))
            eeevals = np.load(os.path.join(trainedexpf['ee'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'ee', mmod, runinfo.planestring())))
            accevals = np.load(os.path.join(trainedexpf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std', runinfo.planestring())))
            labevals = np.load(os.path.join(trainedexpf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
            
            trainedlayerevals = []
            trainedlayerevals.append(dvevals[...,1,1]) #dir
            trainedlayerevals.append(dvevals[...,2,1]) #vel
            trainedlayerevals.append(dvevals[...,3,1]) #dir + vel  
            trainedlayerevals.append(accevals[...,2,1]) #acc
            trainedlayerevals.append(labevals[...,0]) #labels
            trainedlayerevals.append(eeevals[...,0,1]) #ee cart
            trainedlayerevals.append(eeevals[...,3,1]) #ee polar  

            for i in range(len(trainedlayerevals)):
                trainedlayerevals[i][(trainedlayerevals[i] == 1) & (trainedlayerevals[i] > -0.1)] = np.nan

            dvevals = np.load(os.path.join(controlexpf['vel'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'vel', mmod, runinfo.planestring())))
            eeevals = np.load(os.path.join(controlexpf['ee'], 'l%d_%s_mets_%s_%s_test.npy' %( ilayer, 'ee', mmod, runinfo.planestring())))
            accevals = np.load(os.path.join(controlexpf['acc'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'acc', 'std', runinfo.planestring())))
            labevals = np.load(os.path.join(controlexpf['labels'], 'l%d_%s_mets_%s_%s_test.npy' %(ilayer, 'labels', 'std', runinfo.planestring())))
            
            controllayerevals = []
            controllayerevals.append(dvevals[...,1,1]) #dir
            controllayerevals.append(dvevals[...,2,1]) #vel
            controllayerevals.append(dvevals[...,3,1]) #dir + vel
            controllayerevals.append(accevals[...,2,1]) #acc
            controllayerevals.append(labevals[...,0]) #labels 
            controllayerevals.append(eeevals[...,0,1]) #ee cart
            controllayerevals.append(eeevals[...,3,1]) #ee polar  

            for i in range(len(controllayerevals)):
                controllayerevals[i][(controllayerevals[i] == 1) & (controllayerevals[i]> -0.1)] = np.nan

            #print(tcnames)
            for itc, tc in enumerate(tcnames):
                for testtype in testtypes:
                #testtype='two-sided'
                    #print(itc, len(trainedlayerevals))
                    pv = ttest_rel(trainedlayerevals[itc].flatten(), controllayerevals[itc].flatten(), nan_policy='omit', alternative=testtype)[1]

                    print("mname %s , tc %s , ilayer %d , testtype %s , p-value %f" %(trainedmodel['name'], tc, ilayer, testtype, pv))
                    df.loc[(trainedmodel['name'], tc), (ilayer, testtype, 'p-value')] = pv
                    df.loc[(trainedmodel['name'], tc), (ilayer, testtype, 'sl')] = pv_to_sl_code(pv)
                    
                    #df.loc[(trainedmodel['name'], tc), (ilayer, testtypes[1], 'p-value')] = trainedlayerevals[itc].mean()
                    #df.loc[(trainedmodel['name'], tc), (ilayer, testtypes[1], 'sl')] = controllayerevals[itc].mean()
        
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

    linewidth=5.1
        
    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = fig.add_subplot(111)
        
    for i in range(len(tmdevs)):
        #plt.plot(range(len(layers)), tmdevs[i], color=trainedmodel['color'], marker = 'D', alpha = 0.15, label='ind trained')
        plt.plot(range(len(layers)), tmdevs[i], color=trainedmodel['color'], marker = 'D', alpha = 0.15, label='ind trained', linewidth=linewidth)
        #plt.plot(range(len(layers)), cmdevs[i], color='grey', marker = 'D', alpha = 0.15, label='ind control')
        plt.plot(range(len(layers)), cmdevs[i], color='grey', marker = 'D', alpha = 0.15, label='ind untrained', linewidth=linewidth) ## eLife
    
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
    
    plt.plot(range(len(layers)), tmsmean, color=trainedmodel['color'], marker = 'D', linewidth=linewidth)
    plt.plot(range(len(layers)), cmsmean, color='grey', marker = 'D', linewidth=linewidth)
        
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

    ## eLife Style
    ## SET AXIS WIDTHS
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(linewidth)
    # increase tick width
    ax.tick_params(width=linewidth)

    
    #plt.legend(handles[[0,1,10,11]], ['ind trained', 'ind control', \
    #            'mean of trained', 'mean of controls'])
    plt.legend(handles[[0,1,10,11]], ['Ind. Trained', 'Ind. Untrained', \
                'Mean Trained', 'Mean Untrained']) ## eLife
    
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
    #df.to_csv(os.path.join(analysisfolder, 'pd_deviation_normalized.csv'))    
    df.to_csv(os.path.join(analysisfolder, 'pd_deviation.csv'))    
    
    fig = plot_pd_deviation(layers, trainedmodeldevs, controlmodeldevs, model)
    #fig.savefig(os.path.join(analysisfolder, 'pd_dev_plot_normalized.pdf'))
    fig.savefig(os.path.join(analysisfolder, 'pd_dev_plot.pdf'), dpi=300, transparent=True)
    fig.savefig(os.path.join(analysisfolder, 'pd_dev_plot.svg'), dpi=300, transparent=True)
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

def combined_colorselector(cmapname, tcf, ct = 0.4):
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
    
    if tcf != 'acc' and tcf != 'dirvel':
        tcnames = ['dir', 'vel', 'ee', 'eepolar', 'labels']
    else:
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

def colorselector_dec(cmapname, tcf, ct=0.4):
    ''' Helper function to select color for positional tuning curve types and labels for decoding,
    making use of the encoding colorselector functions
    
    Arguments
    ---------
    cmapname : str, the name of the color map that is being used 
    tcf : str, name of tuning feature that is being plotted
    ct : int, opt, specifies range of colorbar that is used
    
    Returns
    -------
    color values
    '''

    if 'ee' in tcf and 'polar' not in tcf:
        color = colorselector_ee(cmapname, 'ee', ct)
    elif 'eepolar' in tcf: 
        color = colorselector_ee(cmapname, 'eepolar', ct)  
    elif 'acc' in tcf:
        color = colorselector(cmapname, 'acc', ct)
    else:
        color = colorselector(cmapname, tcf, ct)
    
    return color

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
    #plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
    #           horizontalalignment = 'right')
    plt.ylim((-0.1,1))
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    plt.legend(['Dir Trained', 'Dir Untrained', \
                'Acc Trained', 'Acc Untrained'])
    
    ax = format_axis(ax)
    
    plt.tight_layout()

    return fig

def plotcomp_tr_reg_dir_accs(tcfdf, tcf, model, regressionmodel):
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
    
    controlnamer = lambda i: regressionmodel['base'] + '_%d' %i
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
    plt.errorbar(x, controldirs, yerr=errs_controldirs, color=colorselector(regressionmodel['regression_cmap'], 'dir'), marker='D', capsize=3.0)
    plt.errorbar(x, trainedlabels, yerr=errs_trainedlabels, color=colorselector(model['cmap'], 'acc'), linestyle='-.',marker='D', capsize=3.0)
    plt.errorbar(x, controllabels, yerr=errs_controllabels, color=colorselector(regressionmodel['regression_cmap'], 'acc'), linestyle='-.', marker='D', capsize=3.0)
    plt.ylabel('r2 score')
    #plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
    #           horizontalalignment = 'right')
    plt.xticks(np.array(x), ['Sp.'] + ['L%d' %i for i in np.arange(1,model['nlayers']+1)])
    plt.ylim((-0.1,1))
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    plt.legend(['Dir Task', 'Dir Decoding', \
                'Acc Task', 'Acc Decoding'])
    
    ax = format_axis(ax)
    
    plt.tight_layout()

    return fig


def plotcomp_tr_reg_twovars(tcfdf, tcfs, model, regressionmodel):
    ''' Plot comparisons between 90% quantiles for trained and control models for direction and acceleration 
    on a single plot
    
    Arguments
    ---------
    tcfdf : pd.DataFrame, index and columns matching output of compile_comparisons
    tcfs : list of strs, name of tuning feature
    model : dict
    
    Returns
    -------
    fig : plt.figure
    '''
    
    #fig = plt.figure(figsize=(12,5.5), dpi=300)   
    fig = plt.figure(figsize=(8,6), dpi=300)   
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_axisbelow(True)
    
    trainednamer = lambda i: model['base'] + '_%d' %i
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    
    controlnamer = lambda i: regressionmodel['base'] + '_%d' %i
    controlnames = [controlnamer(i) for i in np.arange(1,6)]
    
    nlayers = model['nlayers']
    
    x = range(nlayers + 1)

    #solution to calculate conf. interval of means from https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html
    t_corr = t.ppf(0.975, 4)
    
    print(tcfdf.head())

    traineddirs = np.vstack([tcfdf.loc[(trainednames, i, tcfs[0])] for i in np.arange(nlayers+1)]).T
    controldirs = np.vstack([tcfdf.loc[(controlnames, i, tcfs[0])] for i in np.arange(nlayers+1)]).T           
    
    mean_traineddirs = [tcfdf.loc[(trainednames, i, tcfs[0])].mean() for i in np.arange(nlayers+1)]
    mean_controldirs = [tcfdf.loc[(controlnames, i, tcfs[0])].mean() for i in np.arange(nlayers+1)]
    errs_traineddirs = [tcfdf.loc[(trainednames, i, tcfs[0])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controldirs = [tcfdf.loc[(controlnames, i, tcfs[0])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    trainedlabels = np.vstack([tcfdf.loc[(trainednames, i, tcfs[1])] for i in np.arange(nlayers+1)]).T
    controllabels = np.vstack([tcfdf.loc[(controlnames, i, tcfs[1])] for i in np.arange(nlayers+1)]).T

    mean_trainedlabels = [tcfdf.loc[(trainednames, i, tcfs[1])].mean() for i in np.arange(nlayers+1)]
    mean_controllabels = [tcfdf.loc[(controlnames, i, tcfs[1])].mean() for i in np.arange(nlayers+1)]
    errs_trainedlabels = [tcfdf.loc[(trainednames, i, tcfs[1])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controllabels = [tcfdf.loc[(controlnames, i, tcfs[1])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]

    for i, mname in enumerate(trainednames):
        plt.plot(x, traineddirs[i], color=combined_colorselector(model['cmap'], tcfs[0]), marker = 'o', alpha = 0.15, label='ind trained')
        #plt.plot(x, controldirs[i], color=combined_colorselector(regressionmodel['regression_cmap'], tcfs[0]), marker = 'D', alpha = 0.15, label='ind control')
        plt.plot(x, controldirs[i], color=combined_colorselector(regressionmodel['regression_cmap'], tcfs[0]), marker = 'D', alpha = 0.15, label='ind untrained') ##eLife
        plt.plot(x, trainedlabels[i], color=combined_colorselector(model['cmap'], tcfs[1]), linestyle=(0,(5,5)), marker = 'o', alpha = 0.15, label='ind trained')
        #plt.plot(x, controllabels[i], color=combined_colorselector(regressionmodel['regression_cmap'], tcfs[1]), linestyle=(0,(5,5)), marker = 'D', alpha = 0.15, label='ind control')
        plt.plot(x, controllabels[i], color=combined_colorselector(regressionmodel['regression_cmap'], tcfs[1]), linestyle=(0,(5,5)), marker = 'D', alpha = 0.15, label='ind untrained') ##eLife

    #print(traineddirs)
    #print(errs_traineddirs)
    
    plt.errorbar(x, mean_traineddirs, yerr=errs_traineddirs, color=combined_colorselector(model['cmap'], tcfs[0]), marker='o', capsize=3.0, label='mean trained dir')
    plt.errorbar(x, mean_controldirs, yerr=errs_controldirs, color=combined_colorselector(regressionmodel['regression_cmap'], tcfs[0]), marker='D', capsize=3.0, label='mean controls dir')
    plt.errorbar(x, mean_trainedlabels, yerr=errs_trainedlabels, color=combined_colorselector(model['cmap'], tcfs[1]), linestyle=(0,(5,5)), marker='o', capsize=3.0, label='mean trained pos')
    plt.errorbar(x, mean_controllabels, yerr=errs_controllabels, color=combined_colorselector(regressionmodel['regression_cmap'], tcfs[1]), linestyle=(0,(5,5)), marker='D', capsize=3.0, label='mean controls dir')
    plt.ylabel('r2 score')

    #plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
    #           horizontalalignment = 'right')
    plt.xticks(np.array(x), ['Sp.'] + ['L%d' %i for i in np.arange(1,model['nlayers']+1)])
    plt.ylim((-0.1,1))
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)

    #print(handles)

    #leg1 = plt.legend(handles[[20, 21, 22, 23]], ['%s ART' %tcnames_fancy[tcfs[0]], '%s TDT' %tcnames_fancy[tcfs[0]], \
    #            '%s ART' %tcnames_fancy[tcfs[1]], '%s TDT' %tcnames_fancy[tcfs[1]]])

    #plt.legend(handles[[0,1,10,11]], ['Ind Trained', 'Ind Dec', \
    #        'mean of trained', 'mean of controls'])
    
    #plt.legend(handles[[0,20]], ['Ind.', 'Mean'], loc='upper left')

    print("NEW LEGEND YAY")

    leg1 = plt.legend(handles[[0, 20, 2, 22]], ['%s Ind.' %tcnames_fancy[tcfs[0]], '%s Mean' %tcnames_fancy[tcfs[0]], \
                '%s Ind.' %tcnames_fancy[tcfs[1]], '%s Mean' %tcnames_fancy[tcfs[1]]])

    #plt.legend(handles[[0,1,10,11]], ['Ind Trained', 'Ind Dec', \
    #        'mean of trained', 'mean of controls'])
    
    #plt.legend(handles[[0,20]], ['Ind.', 'Mean'], loc='upper left')
    #print("MODEL REGRESSION TASK for model %s : %s" %(model['name'], model['regression_task']))
    plt.legend(handles[[20,21]], ['ART', 'TDT'], loc='upper left')

    ax.add_artist(leg1)

    ax = format_axis(ax)
    
    plt.tight_layout()

    return fig

def plotcomp_twovars(tcfdf, tcfs, model):
    ''' Plot comparisons between 90% quantiles for trained and control models for direction and acceleration 
    on a single plot
    
    Arguments
    ---------
    tcfdf : pd.DataFrame, index and columns matching output of compile_comparisons
    tcfs : list of strs, name of tuning feature
    model : dict
    
    Returns
    -------
    fig : plt.figure
    '''
    
    #fig = plt.figure(figsize=(12,5.5), dpi=300)   
    fig = plt.figure(figsize=(8,6), dpi=300)   
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

    traineddirs = np.vstack([tcfdf.loc[(trainednames, i, tcfs[0])] for i in np.arange(nlayers+1)]).T
    controldirs = np.vstack([tcfdf.loc[(controlnames, i, tcfs[0])] for i in np.arange(nlayers+1)]).T           
    
    mean_traineddirs = [tcfdf.loc[(trainednames, i, tcfs[0])].mean() for i in np.arange(nlayers+1)]
    mean_controldirs = [tcfdf.loc[(controlnames, i, tcfs[0])].mean() for i in np.arange(nlayers+1)]
    errs_traineddirs = [tcfdf.loc[(trainednames, i, tcfs[0])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controldirs = [tcfdf.loc[(controlnames, i, tcfs[0])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    trainedlabels = np.vstack([tcfdf.loc[(trainednames, i, tcfs[1])] for i in np.arange(nlayers+1)]).T
    controllabels = np.vstack([tcfdf.loc[(controlnames, i, tcfs[1])] for i in np.arange(nlayers+1)]).T

    mean_trainedlabels = [tcfdf.loc[(trainednames, i, tcfs[1])].mean() for i in np.arange(nlayers+1)]
    mean_controllabels = [tcfdf.loc[(controlnames, i, tcfs[1])].mean() for i in np.arange(nlayers+1)]
    errs_trainedlabels = [tcfdf.loc[(trainednames, i, tcfs[1])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controllabels = [tcfdf.loc[(controlnames, i, tcfs[1])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]

    for i, mname in enumerate(trainednames):
        plt.plot(x, traineddirs[i], color=combined_colorselector(model['cmap'], tcfs[0]), marker = 'o', alpha = 0.15, label='ind trained')
        #plt.plot(x, controldirs[i], color=combined_colorselector('Greys_r', tcfs[0]), marker = 'D', alpha = 0.15, label='ind control')
        plt.plot(x, controldirs[i], color=combined_colorselector('Greys_r', tcfs[0]), marker = 'D', alpha = 0.15, label='ind untrained') ## eLife
        plt.plot(x, trainedlabels[i], color=combined_colorselector(model['cmap'], tcfs[1]), linestyle=(0,(5,5)), marker = 'o', alpha = 0.15, label='ind trained')
        ##plt.plot(x, controllabels[i], color=combined_colorselector('Greys_r', tcfs[1]), linestyle=(0,(5,5)), marker = 'D', alpha = 0.15, label='ind control')
        plt.plot(x, controllabels[i], color=combined_colorselector('Greys_r', tcfs[1]), linestyle=(0,(5,5)), marker = 'D', alpha = 0.15, label='ind untrained')

    #print(traineddirs)
    #print(errs_traineddirs)
    
    plt.errorbar(x, mean_traineddirs, yerr=errs_traineddirs, color=combined_colorselector(model['cmap'], tcfs[0]), marker='o', capsize=3.0, label='mean trained dir')
    plt.errorbar(x, mean_controldirs, yerr=errs_controldirs, color=combined_colorselector('Greys_r', tcfs[0]), marker='D', capsize=3.0, label='mean untrained dir')
    plt.errorbar(x, mean_trainedlabels, yerr=errs_trainedlabels, color=combined_colorselector(model['cmap'], tcfs[1]), linestyle=(0,(5,5)), marker='o', capsize=3.0, label='mean trained pos')
    plt.errorbar(x, mean_controllabels, yerr=errs_controllabels, color=combined_colorselector('Greys_r', tcfs[1]), linestyle=(0,(5,5)), marker='D', capsize=3.0, label='mean untrained dir')
    plt.ylabel('r2 score')

    #plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
    #           horizontalalignment = 'right')
    plt.xticks(np.array(x), ['Sp.'] + ['L%d' %i for i in np.arange(1,model['nlayers']+1)])
    plt.ylim((-0.1,1))
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)

    #print(handles)

    #leg1 = plt.legend(handles[[20, 21, 22, 23]], ['%s ART' %tcnames_fancy[tcfs[0]], '%s TDT' %tcnames_fancy[tcfs[0]], \
    #            '%s ART' %tcnames_fancy[tcfs[1]], '%s TDT' %tcnames_fancy[tcfs[1]]])

    leg1 = plt.legend(handles[[0, 20, 2, 22]], ['%s Ind.' %tcnames_fancy[tcfs[0]], '%s Mean' %tcnames_fancy[tcfs[0]], \
                '%s Ind.' %tcnames_fancy[tcfs[1]], '%s Mean' %tcnames_fancy[tcfs[1]]])

    #plt.legend(handles[[0,1,10,11]], ['Ind Trained', 'Ind Dec', \
    #        'mean of trained', 'mean of controls'])
    
    #plt.legend(handles[[0,20]], ['Ind.', 'Mean'], loc='upper left')
    print("MODEL REGRESSION TASK for model %s : %s" %(model['name'], model['regression_task']))
    if not model['regression_task']:
        plt.legend(handles[[20,21]], ['ART-trained', 'Untrained'], loc='upper left')
    else:
        plt.legend(handles[[20,21]], ['TDT-trained', 'Untrained'], loc='upper left')

    ax.add_artist(leg1)

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
    #plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
    #           horizontalalignment = 'right')\
    plt.xticks(np.array(x), ['Sp.'] + ['L%d' %i for i in np.arange(1,model['nlayers']+1)])
    plt.ylim((-0.1,1))
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    plt.legend(['Cart Trained', 'Cart Untrained', \
                'Polar Trained', 'Polar Untrained'])
    
    ax = format_axis(ax)
    
    plt.tight_layout()

    return fig

def plotcomp_tr_reg_ees(tcfdf, model, regressionmodel):
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
    
    controlnamer = lambda i: regressionmodel['base'] + '_%d' %i
    controlnames = [controlnamer(i) for i in np.arange(1,6)]
    
    nlayers = model['nlayers']
    
    x = range(nlayers + 1)
        
    #solution to calculate conf. interval of means from https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html
    t_corr = t.ppf(0.975, 4)
    
    print(tcfdf.head())
    print(tcfdf.shape)
    #print([tcfdf.loc[(trainednames, i, 'ee')])
    traineddirs = [np.nanmean(tcfdf.loc[(trainednames, i, 'ee')],dtype='float32') for i in np.arange(nlayers+1)]
    controldirs = [np.nanmean(tcfdf.loc[(controlnames, i, 'ee')],dtype='float32') for i in np.arange(nlayers+1)]
    errs_traineddirs = [np.nanstd(tcfdf.loc[(trainednames, i, 'ee')],dtype='float32')/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controldirs = [np.nanstd(tcfdf.loc[(controlnames, i, 'ee')],dtype='float32')/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    trainedlabels = [np.nanmean(tcfdf.loc[(trainednames, i, 'eepolar')],dtype='float32') for i in np.arange(nlayers+1)]
    controllabels = [np.nanmean(tcfdf.loc[(controlnames, i, 'eepolar')],dtype='float32') for i in np.arange(nlayers+1)]
    errs_trainedlabels = [np.nanstd(tcfdf.loc[(trainednames, i, 'eepolar')],dtype='float32')/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controllabels = [np.nanstd(tcfdf.loc[(controlnames, i, 'eepolar')],dtype='float32')/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    print(traineddirs)
    print(errs_traineddirs)
    
    plt.errorbar(x, traineddirs, yerr=errs_traineddirs, color=colorselector_ee(model['cmap'], 'ee'), marker='D', capsize=3.0)
    plt.errorbar(x, controldirs, yerr=errs_controldirs, color=colorselector_ee(regressionmodel['regression_cmap'], 'ee'), marker='D', capsize=3.0)
    plt.errorbar(x, trainedlabels, yerr=errs_trainedlabels, color=colorselector_ee(model['cmap'], 'eepolar'), linestyle=(0,(5,5)), marker='D', capsize=3.0)
    plt.errorbar(x, controllabels, yerr=errs_controllabels, color=colorselector_ee(regressionmodel['regression_cmap'], 'eepolar'), linestyle=(0,(5,5)), marker='D', capsize=3.0)
        
    plt.ylabel('r2 score')
    #plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
    #           horizontalalignment = 'right')
    plt.xticks(np.array(x), ['Sp.'] + ['L%d' %i for i in np.arange(1,model['nlayers']+1)])
    plt.ylim((-0.1,1))
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    plt.legend(['Cart Task', 'Cart Decoding', \
                'Polar Task', 'Polar Decoding'])
    
    ax = format_axis(ax)
    
    plt.tight_layout()

    return fig

def plotcomp_decoding(tcfdf, tcf, model):
    ''' Plot comparisons between for trained and control models for decoding
    
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
    
    traineddirs = np.vstack([tcfdf.loc[(trainednames, i, tcf)] for i in np.arange(nlayers+1)]).T
    controldirs = np.vstack([tcfdf.loc[(controlnames, i, tcf)] for i in np.arange(nlayers+1)]).T

    traineddirs_mean = [tcfdf.loc[(trainednames, i, tcf)].mean() for i in np.arange(nlayers+1)]
    controldirs_mean = [tcfdf.loc[(controlnames, i, tcf)].mean() for i in np.arange(nlayers+1)]
    errs_traineddirs = [tcfdf.loc[(trainednames, i, tcf)].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controldirs = [tcfdf.loc[(controlnames, i, tcf)].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    #print(traineddirs)
    #print(traineddirs_mean)

    for i, mname in enumerate(trainednames):
        plt.plot(x, traineddirs[i], color=model['color'], marker = 'D', alpha = 0.15, label='ind trained')
        #plt.plot(x, controldirs[i], color='grey', marker = 'D', alpha = 0.15, label='ind control')
        plt.plot(x, controldirs[i], color='grey', marker = 'D', alpha = 0.15, label='ind untrained') ##eLife
    
    plt.errorbar(x, traineddirs_mean, yerr=errs_traineddirs, color=colorselector_dec(model['cmap'], tcf), marker='D', capsize=3.0, label='mean trained')
    #plt.errorbar(x, controldirs_mean, yerr=errs_controldirs, color=colorselector_dec('Greys_r', tcf), marker='D', capsize=3.0, label='mean controls')
    plt.errorbar(x, controldirs_mean, yerr=errs_controldirs, color=colorselector_dec('Greys_r', tcf), marker='D', capsize=3.0, label='mean untrained') ##eLife
    plt.ylabel('r2 score')
    #plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
    #           horizontalalignment = 'right')
    plt.xticks(np.array(x), ['Sp.'] + ['L%d' %i for i in np.arange(1,model['nlayers']+1)])
    plt.ylim((-0.1,1))
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    plt.legend(['%s trained' %tcf, '%s controls' %tcf])
    
    ax = plt.gca()
    format_axis(ax)
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    
    #plt.legend(handles[[0,1,10,11]], ['ind trained', 'ind control', \
    #            'mean of trained', 'mean of controls'])

    plt.legend(handles[[0,1,10,11]], ['ind trained', 'ind untrained', \
                'mean of trained', 'mean of untrained']) ## eLife
    
    plt.tight_layout()

    return fig


def plotcomp_decoding_twovars(tcfdf, tcfs, model):
    ''' Plot comparisons between for trained and control models for decoding
    
    Arguments
    ---------
    tcfdf : pd.DataFrame, index and columns matching output of compile_comparisons
    tcf : str, name of tuning feature
    model : dict
    
    Returns
    -------
    fig : plt.figure
    '''
    
    fig = plt.figure(figsize=(8,6), dpi=300)   
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
    
    trainedvars1 = np.vstack([tcfdf.loc[(trainednames, i, tcfs[0])] for i in np.arange(nlayers+1)]).T
    controlvars1 = np.vstack([tcfdf.loc[(controlnames, i, tcfs[0])] for i in np.arange(nlayers+1)]).T

    trainedvars1_mean = [tcfdf.loc[(trainednames, i, tcfs[0])].mean() for i in np.arange(nlayers+1)]
    controlvars1_mean = [tcfdf.loc[(controlnames, i, tcfs[0])].mean() for i in np.arange(nlayers+1)]
    errs_trainedvars1 = [tcfdf.loc[(trainednames, i, tcfs[0])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controlvars1 = [tcfdf.loc[(controlnames, i, tcfs[0])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]


    trainedvars2 = np.vstack([tcfdf.loc[(trainednames, i, tcfs[1])] for i in np.arange(nlayers+1)]).T
    controlvars2 = np.vstack([tcfdf.loc[(controlnames, i, tcfs[1])] for i in np.arange(nlayers+1)]).T

    trainedvars2_mean = [tcfdf.loc[(trainednames, i, tcfs[1])].mean() for i in np.arange(nlayers+1)]
    controlvars2_mean = [tcfdf.loc[(controlnames, i, tcfs[1])].mean() for i in np.arange(nlayers+1)]
    errs_trainedvars2 = [tcfdf.loc[(trainednames, i, tcfs[1])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controlvars2 = [tcfdf.loc[(controlnames, i, tcfs[1])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]

    for i, mname in enumerate(trainednames):
        line_indtrainedvar1, = plt.plot(x, trainedvars1[i], color=colorselector_dec(model['cmap'], tcfs[0]), marker = 'o', alpha = 0.15, label='Ind. ART')
        line_indcontrolsvar1, = plt.plot(x, controlvars1[i], color=colorselector_dec('Greys_r', tcfs[0]), marker = 'D', alpha = 0.15, label='Ind. TDT')
        line_indtrainedvar2, =plt.plot(x, trainedvars2[i], color=colorselector_dec(model['cmap'], tcfs[1]), marker = 'o', linestyle=(0,(5,5)), alpha = 0.15, label='Ind. ART')
        line_indcontrolsvar2, = plt.plot(x, controlvars2[i], color=colorselector_dec('Greys_r', tcfs[1]), marker = 'D',linestyle=(0,(5,5)), alpha = 0.15, label='Ind. TDT')
    
    line_meantrainedvar1,_,_ = plt.errorbar(x, trainedvars1_mean, yerr=errs_trainedvars1, color=colorselector_dec(model['cmap'], tcfs[0]), marker='o', capsize=3.0, label='Mean ART')
    line_meancontrolsvar1,_,_ = plt.errorbar(x, controlvars1_mean, yerr=errs_controlvars1, color=colorselector_dec('Greys_r', tcfs[0]), marker='D', capsize=3.0, label='Mean TDT')
    line_meantrainedvar2,_,_ = plt.errorbar(x, trainedvars2_mean, yerr=errs_trainedvars2, color=colorselector_dec(model['cmap'], tcfs[1]), marker='o', linestyle=(0,(5,5)), capsize=3.0, label='Mean ART')
    line_meancontrolsvar2,_,_ = plt.errorbar(x, controlvars2_mean, yerr=errs_controlvars2, color=colorselector_dec('Greys_r', tcfs[1]), marker='D', linestyle=(0,(5,5)), capsize=3.0, label='Mean TDT')

    plt.ylabel('r2 score')
    #plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
    #           horizontalalignment = 'right')
    plt.xticks(np.array(x), ['Sp.'] + ['L%d' %i for i in np.arange(1,model['nlayers']+1)])
    plt.ylim((-0.1,1))

    first_legend = plt.legend(handles=[line_indtrainedvar1, line_meantrainedvar1, line_indtrainedvar2, line_meantrainedvar2], \
        labels=['Ind. %s ' %dec_tcnames_fancy[tcfs[0]], 'Mean %s ' %dec_tcnames_fancy[tcfs[0]], 'Ind. %s' %dec_tcnames_fancy[tcfs[1]], 'Mean %s' %dec_tcnames_fancy[tcfs[1]] ], loc="upper right")
    ax.add_artist(first_legend)

    #plt.legend(handles=[line_indtrainedvar1, line_meantrainedvar1], labels=["Ind.", 'Mean'], loc='upper left')
    if not model['regression_task']:
        #plt.legend(handles=[line_meantrainedvar1, line_meancontrolsvar1], labels=["ART-trained", 'Untrained'], loc='upper left')
        plt.legend(handles=[line_meantrainedvar1, line_meancontrolsvar1], labels=["ART-trained", 'Untrained'], loc='upper left') ##eLife
    else:
        #plt.legend(handles=[line_meantrainedvar1, line_meancontrolsvar1], labels=["TDT-trained", 'Untrained'], loc='upper left')
        plt.legend(handles=[line_meantrainedvar1, line_meancontrolsvar1], labels=["TDT-trained", 'Untrained'], loc='upper left')

    #print("Handles: ", handles)
    
    #plt.legend(handles[[0,1,10,11]], ['Ind. Recog.', 'Ind. Decod.', \
    #            'Mean of Recog.', 'Mean of Decod.'])
    
    plt.tight_layout()

    return fig


def plotcomp_tr_reg_decoding(tcfdf, tcf, model, regressionmodel):
    ''' Plot comparisons between for trained and control models for decoding
    
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
    
    controlnamer = lambda i: regressionmodel['base'] + '_%d' %i
    controlnames = [controlnamer(i) for i in np.arange(1,6)]
    
    nlayers = model['nlayers']
    
    x = range(nlayers + 1)

    #solution to calculate conf. interval of means from https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html
    t_corr = t.ppf(0.975, 4)
    
    print(tcfdf.head())
    
    traineddirs = np.vstack([tcfdf.loc[(trainednames, i, tcf)] for i in np.arange(nlayers+1)]).T
    controldirs = np.vstack([tcfdf.loc[(controlnames, i, tcf)] for i in np.arange(nlayers+1)]).T

    traineddirs_mean = [tcfdf.loc[(trainednames, i, tcf)].mean() for i in np.arange(nlayers+1)]
    controldirs_mean = [tcfdf.loc[(controlnames, i, tcf)].mean() for i in np.arange(nlayers+1)]
    errs_traineddirs = [tcfdf.loc[(trainednames, i, tcf)].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controldirs = [tcfdf.loc[(controlnames, i, tcf)].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    #print(traineddirs)
    #print(traineddirs_mean)

    for i, mname in enumerate(trainednames):
        plt.plot(x, traineddirs[i], color=model['color'], marker = 'D', alpha = 0.15, label='ind task')
        plt.plot(x, controldirs[i], color=regressionmodel['regression_color'], marker = 'D', alpha = 0.15, label='ind decoding')
    
    plt.errorbar(x, traineddirs_mean, yerr=errs_traineddirs, color=colorselector_dec(model['cmap'], tcf), marker='D', capsize=3.0, label='mean trained')
    plt.errorbar(x, controldirs_mean, yerr=errs_controldirs, color=colorselector_dec(regressionmodel['regression_cmap'], tcf), marker='D', capsize=3.0, label='mean controls')
    plt.ylabel('r2 score')
    #plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
    #           horizontalalignment = 'right')
    plt.xticks(np.array(x), ['Sp.'] + ['L%d' %i for i in np.arange(1,model['nlayers']+1)])
    plt.ylim((-0.1,1))
    
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    plt.legend(['%s trained' %tcf, '%s controls' %tcf])
    
    ax = plt.gca()
    format_axis(ax)
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    
    plt.legend(handles[[0,1,10,11]], ['Ind. Recog.', 'Ind. Decod.', \
                'Mean of Recog.', 'Mean of Decod.'])
    
    plt.tight_layout()

    return fig

def plotcomp_tr_reg_decoding_twovars(tcfdf, tcfs, model, regressionmodel):
    ''' Plot comparisons between for trained and control models for decoding
    
    Arguments
    ---------
    tcfdf : pd.DataFrame, index and columns matching output of compile_comparisons
    tcfs : list of strs, names of tuning features
    model : dict
    regressionmodel : dict
    
    Returns
    -------
    fig : plt.figure
    '''
    
    #fig = plt.figure(figsize=(12,5.5), dpi=300)   
    fig = plt.figure(figsize=(9,5), dpi=300)   
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_axisbelow(True)
    
    trainednamer = lambda i: model['base'] + '_%d' %i
    trainednames = [trainednamer(i) for i in np.arange(1,6)]
    
    controlnamer = lambda i: regressionmodel['base'] + '_%d' %i
    controlnames = [controlnamer(i) for i in np.arange(1,6)]
    
    nlayers = model['nlayers']
    
    x = range(nlayers + 1)

    #solution to calculate conf. interval of means from https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html
    t_corr = t.ppf(0.975, 4)
    
    print(tcfdf.head())
    
    trainedvars1 = np.vstack([tcfdf.loc[(trainednames, i, tcfs[0])] for i in np.arange(nlayers+1)]).T
    controlvars1 = np.vstack([tcfdf.loc[(controlnames, i, tcfs[0])] for i in np.arange(nlayers+1)]).T

    trainedvars2 = np.vstack([tcfdf.loc[(trainednames, i, tcfs[1])] for i in np.arange(nlayers+1)]).T
    controlvars2 = np.vstack([tcfdf.loc[(controlnames, i, tcfs[1])] for i in np.arange(nlayers+1)]).T

    trainedvars1_mean = [tcfdf.loc[(trainednames, i, tcfs[0])].mean() for i in np.arange(nlayers+1)]
    controlvars1_mean = [tcfdf.loc[(controlnames, i, tcfs[0])].mean() for i in np.arange(nlayers+1)]
    errs_trainedvars1 = [tcfdf.loc[(trainednames, i, tcfs[0])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controlvars1 = [tcfdf.loc[(controlnames, i, tcfs[0])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    
    trainedvars2_mean = [tcfdf.loc[(trainednames, i, tcfs[1])].mean() for i in np.arange(nlayers+1)]
    controlvars2_mean = [tcfdf.loc[(controlnames, i, tcfs[1])].mean() for i in np.arange(nlayers+1)]
    errs_trainedvars2 = [tcfdf.loc[(trainednames, i, tcfs[1])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]
    errs_controlvars2 = [tcfdf.loc[(controlnames, i, tcfs[1])].std()/np.sqrt(5) * t_corr for i in np.arange(nlayers+1)]

    #print(traineddirs)
    #print(traineddirs_mean)

    for i, mname in enumerate(trainednames):
        line_indtrainedvar1, = plt.plot(x, trainedvars1[i], color=model['color'], marker = 'o', alpha = 0.15, label='Ind. Recog.')
        line_indcontrolsvar1, = plt.plot(x, controlvars1[i], color=regressionmodel['regression_color'], marker = 'D', alpha = 0.15, label='Ind. Decod.')
        line_indtrainedvar2, = plt.plot(x, trainedvars2[i], color=model['color'], marker = 'o', linestyle=(0,(5,5)), alpha = 0.15, label='Ind. Recog.')
        plt.plot(x, controlvars2[i], color=regressionmodel['regression_color'], marker = 'D',linestyle=(0,(5,5)), alpha = 0.15, label='Ind. Decod.')
    
    line_meantrainedvar1,_,_ = plt.errorbar(x, trainedvars1_mean, yerr=errs_trainedvars1, color=colorselector_dec(model['cmap'], tcfs[0]), marker='o', capsize=3.0, label='Mean Recog.')
    line_meancontrolsvar1,_,_ = plt.errorbar(x, controlvars1_mean, yerr=errs_controlvars1, color=colorselector_dec(regressionmodel['regression_cmap'], tcfs[0]), marker='D', capsize=3.0, label='Mean Decod.')
    line_meantrainedvar2,_,_ = plt.errorbar(x, trainedvars2_mean, yerr=errs_trainedvars2, color=colorselector_dec(model['cmap'], tcfs[1]), marker='o', linestyle=(0,(5,5)), capsize=3.0, label='Mean Recog.')
    line_meancontrolsvar2,_,_ = plt.errorbar(x, controlvars2_mean, yerr=errs_controlvars2, color=colorselector_dec(regressionmodel['regression_cmap'], tcfs[1]), marker='D', linestyle=(0,(5,5)), capsize=3.0, label='Mean Decod.')

    plt.ylabel('r2 score')
    #plt.xticks(np.array(x), ['Spindles'] + ['Layer %d' %i for i in np.arange(1,model['nlayers']+1)], rotation=45,
    #           horizontalalignment = 'right')
    plt.xticks(np.array(x), ['Sp.'] + ['L%d' %i for i in np.arange(1,model['nlayers']+1)])
    plt.ylim((-0.1,1))
    
    #handles, _ = ax.get_legend_handles_labels()
    #handles = np.array(handles)
    #plt.legend(['%s trained' %tcf, '%s controls' %tcf])
    
    #ax = plt.gca()
    #format_axis(ax)
    #handles, _ = ax.get_legend_handles_labels()
    #handles = np.array(handles)

    #first_legend = plt.legend(handles=[line_meantrainedvar1, line_meancontrolsvar1, line_meantrainedvar2, line_meancontrolsvar2], \
    #    labels=['%s Recog.' %dec_tcnames_fancy[tcfs[0]], '%s Decod.' %dec_tcnames_fancy[tcfs[0]], '%s Recog.' %dec_tcnames_fancy[tcfs[1]], '%s Decod.' %dec_tcnames_fancy[tcfs[1]] ], loc="upper right")
    #ax.add_artist(first_legend)

    #plt.legend(handles=[line_indtrainedvar1, line_meantrainedvar1], labels=["Ind.", 'Mean'], loc='upper left')

    #print("Handles: ", handles)
    
    #plt.legend(handles[[0,1,10,11]], ['Ind. Recog.', 'Ind. Decod.', \
    #            'Mean of Recog.', 'Mean of Decod.'])

    first_legend = plt.legend(handles=[line_indtrainedvar1, line_meantrainedvar1, line_indtrainedvar2, line_meantrainedvar2], \
        labels=['Ind. %s ' %dec_tcnames_fancy[tcfs[0]], 'Mean %s ' %dec_tcnames_fancy[tcfs[0]], 'Ind. %s' %dec_tcnames_fancy[tcfs[1]], 'Mean %s' %dec_tcnames_fancy[tcfs[1]] ], loc="upper right")
    ax.add_artist(first_legend)

    #plt.legend(handles=[line_indtrainedvar1, line_meantrainedvar1], labels=["Ind.", 'Mean'], loc='upper left')
    '''
    if not model['regression_task']:
        plt.legend(handles=[line_meantrainedvar1, line_meancontrolsvar1], labels=["ART-trained", 'Untrained'], loc='upper left')
    else:
        plt.legend(handles=[line_meantrainedvar1, line_meancontrolsvar1], labels=["TDT-trained", 'Untrained'], loc='upper left')
    '''
    plt.legend(handles=[line_meantrainedvar1, line_meancontrolsvar1], labels=["ART", 'TDT'], loc='upper left')
    
    plt.tight_layout()

    format_axis(plt.gca())

    return fig

def decoding_tcctrlcompplots(df, model, runinfo, alpha = None):
    ''' Saves plots comparing the decoding strengths of the model for various tuning feature types
    
    Arguments
    ---------
    df : pd.DataFrame matching output of compile_decoding_comparisons
    model : dict
    runinfo : RunInfo (extension of dict)
    
    '''
    
    folder = runinfo.sharedanalysisfolder(model,  'decoding_kindiffs_plots')
    os.makedirs(folder, exist_ok=True)
    
    for tcname in dec_tcnames:

        tcdf = df.loc[(slice(None), slice(None), tcname), 'r2']#.reset_index(level=2, drop=True)
        
        fig = plotcomp_decoding(tcdf, tcname, model)
        #SWITCH FOR NORMALIZATION
        #fig.savefig(os.path.join(folder, 'decoding_%s_comp_normalized.pdf' %tcname))
        #fig.savefig(os.path.join(folder, 'decoding_%s_comp_normalized.svg' %tcname))
        if alpha is None:
            fig.savefig(os.path.join(folder, 'decoding_%s_comp.pdf' %tcname))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp.svg' %tcname))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp.png' %tcname))
        else:
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_a%d.pdf' %(tcname, int(alpha*1000))))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_a%d.svg' %(tcname, int(alpha*1000))))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_a%d.png' %(tcname, int(alpha*1000))))

        tcdf_RMSE = df.loc[(slice(None), slice(None), tcname), 'RMSE']#.reset_index(level=2, drop=True)
        
        fig = plotcomp_decoding(tcdf_RMSE, tcname, model)
        #SWITCH FOR NORMALIZATION
        #fig.savefig(os.path.join(folder, 'decoding_%s_comp_normalized_RMSE.pdf' %tcname))
        #fig.savefig(os.path.join(folder, 'decoding_%s_comp_normalized_RMSE.svg' %tcname))

        if alpha is None:
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_RMSE.pdf' %tcname))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_RMSE.svg' %tcname))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_RMSE.png' %tcname))
        else:
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_RMSE_a%d.pdf' %(tcname, int(alpha*1000) )))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_RMSE_a%d.svg' %(tcname, int(alpha*1000))))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_RMSE_a%d.png' %(tcname, int(alpha*1000))))

        plt.close('all')

    df_r2 = df.loc[(slice(None), slice(None), slice(None)), 'r2']

    fig = plotcomp_decoding_twovars(df_r2, ['ee_x','ee_y'], model)

    #NORMALIZATION
    #fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_normalized.pdf'))
    #fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_normalized.svg'))
    if alpha is None:
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp.pdf'))
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp.svg'))
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp.png'))
    else:
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_a%d.pdf' %int(alpha*1000)))
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_a%d.png' %int(alpha*1000)))
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_a%d.svg' %int(alpha*1000)))

    fig = plotcomp_decoding_twovars(df_r2, ['dir','vel'], model)
    #NORMALIZATION
    #fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_normalized.pdf'))
    #fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_normalized.svg'))
    if alpha is None:
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp.pdf'))
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp.svg'))
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp.png'))
    else:
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp_a%d.pdf' %int(alpha*1000)))
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp_a%d.svg' %int(alpha*1000)))
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp_a%d.png' %int(alpha*1000)))


def decoding_tcctrlcompplots_tr_reg(df, model, regressionmodel, runinfo, alpha = None):
    ''' Saves plots comparing the decoding strengths of the model for various tuning feature types
    
    Arguments
    ---------
    df : pd.DataFrame matching output of compile_decoding_comparisons
    model : dict
    runinfo : RunInfo (extension of dict)
    
    '''
    
    folder = runinfo.sharedanalysisfolder(model,  'decoding_kindiffs_tr_reg_plots')
    os.makedirs(folder, exist_ok=True)
    
    for tcname in dec_tcnames:

        tcdf = df.loc[(slice(None), slice(None), tcname), 'r2']#.reset_index(level=2, drop=True)
        
        fig = plotcomp_tr_reg_decoding(tcdf, tcname, model, regressionmodel)
        #SWITCH FOR NORMALIZATION
        #fig.savefig(os.path.join(folder, 'decoding_%s_comp_tr_reg_normalized.pdf' %tcname))
        #fig.savefig(os.path.join(folder, 'decoding_%s_comp_tr_reg_normalized.svg' %tcname))
        if alpha is None:
            fig.savefig(os.path.join(folder, 'decoding_%s_comp.pdf' %tcname))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp.svg' %tcname))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp.png' %tcname))
        else:
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_a%d.pdf' %(tcname, int(alpha*1000))))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_a%d.svg' %(tcname, int(alpha*1000))))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_a%d.png' %(tcname, int(alpha*1000))))

        tcdf_RMSE = df.loc[(slice(None), slice(None), tcname), 'RMSE']#.reset_index(level=2, drop=True)
        
        fig = plotcomp_tr_reg_decoding(tcdf_RMSE, tcname, model, regressionmodel)
        #SWITCH FOR NORMALIZATION
        #fig.savefig(os.path.join(folder, 'decoding_%s_comp_tr_reg_normalized_RMSE.pdf' %tcname))
        #fig.savefig(os.path.join(folder, 'decoding_%s_comp_tr_reg_normalized_RMSE.svg' %tcname))
        if alpha is None:
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_RMSE.pdf' %tcname))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_RMSE.svg' %tcname))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_RMSE.png' %tcname))
        else:
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_a%d_RMSE.pdf' %(tcname, int(alpha*1000))))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_a%d_RMSE.svg' %(tcname, int(alpha*1000))))
            fig.savefig(os.path.join(folder, 'decoding_%s_comp_a%d_RMSE.png' %(tcname, int(alpha*1000))))
        
        plt.close('all')

    ##Plot pairs of variables
    df_r2 = df.loc[(slice(None), slice(None), slice(None)), 'r2']

    fig = plotcomp_tr_reg_decoding_twovars(df_r2, ['ee_x','ee_y'], model, regressionmodel)
    #NORMALIZATION
    #fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_tr_reg_normalized.pdf'))
    #fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_tr_reg_normalized.svg'))
    if alpha is None:
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_tr_reg.pdf'))
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_tr_reg.svg'))
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_tr_reg.png'))
    else:
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_tr_reg_a%d.pdf' %int(alpha*1000)))
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_tr_reg_a%d.svg' %int(alpha*1000)))
        fig.savefig(os.path.join(folder, 'decoding_ee_xy_comp_tr_reg_a%d.png' %int(alpha*1000)))

    fig = plotcomp_tr_reg_decoding_twovars(df_r2, ['dir','acc_r'], model, regressionmodel)
    #NORMALIZATION
    #fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_normalized.pdf'))
    #fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_normalized.svg'))
    if alpha is None:
        fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg.pdf'))
        fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg.svg'))
        fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg.png'))
    else:
        fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_a%d.pdf' %int(alpha*1000)))
        fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_a%d.png' %int(alpha*1000)))
        fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_a%d.svg' %int(alpha*1000)))

    fig = plotcomp_tr_reg_decoding_twovars(df_r2, ['dir','vel'], model, regressionmodel)
    #NORMALIZATION
    #fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_normalized.pdf'))
    #fig.savefig(os.path.join(folder, 'decoding_diracc_comp_tr_reg_normalized.svg'))
    if alpha is None:
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp_tr_reg.pdf'))
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp_tr_reg.svg'))
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp_tr_reg.png'))
    else:
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp_tr_reg_a%d.pdf' %int(alpha*1000)))
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp_tr_reg_a%d.svg' %int(alpha*1000)))
        fig.savefig(os.path.join(folder, 'decoding_dirvel_comp_tr_reg_a%d.png' %int(alpha*1000)))


    ###COMBINED DECODING POS AND EE PLOT
    #df_r2['ee_mean'] = df_r2[['ee_x', 'ee_y']].mean(axis=1)
    fig = plotcomp_tr_reg_decoding_twovars(df_r2, ['dir', 'ee_mean'], model, regressionmodel)
    if alpha is None:
        fig.savefig(os.path.join(folder, 'decoding_comp_tr_reg_combined.pdf'))
        fig.savefig(os.path.join(folder, 'decoding_comp_tr_reg_combined.svg'))
        fig.savefig(os.path.join(folder, 'decoding_comp_tr_reg_combined.png'))
    else:
        fig.savefig(os.path.join(folder, 'decoding_comp_tr_reg_combined_a%d.pdf'%int(alpha*1000)), transparent=True, dpi=300)
        fig.savefig(os.path.join(folder, 'decoding_comp_tr_reg_combined_a%d.svg'%int(alpha*1000)), transparen=True, dpi=300)
        fig.savefig(os.path.join(folder, 'decoding_comp_tr_reg_combined_a%d.png'%int(alpha*1000)), transparent=True, dpi=300)

        ## eLife, save in new size
        fig.set_size_inches(8,6)
        fig.savefig(os.path.join(folder, 'decoding_comp_tr_reg_combined_a%d_narrow.pdf'%int(alpha*1000)), transparent=True, dpi=300)
        fig.savefig(os.path.join(folder, 'decoding_comp_tr_reg_combined_a%d_narrow.svg'%int(alpha*1000)), transparen=True, dpi=300)
        fig.savefig(os.path.join(folder, 'decoding_comp_tr_reg_combined_a%d_narrow.png'%int(alpha*1000)), transparent=True, dpi=300)

def tcctrlcompplots_tr_reg(df, model, regressionmodel, runinfo):
    ''' Saves plots comparing 90% quantiles for the model for various tuning feature types
    
    Arguments
    ---------
    df : pd.DataFrame matching output of compile_comparisons
    model : dict
    runinfo : RunInfo (extension of dict)
    
    '''

    print('TCCTRLCOMPPLOTS TR_REG')
    
    folder = runinfo.sharedanalysisfolder(model,  'kindiffs_tr_reg_plots')
    os.makedirs(folder, exist_ok=True)
    tcf=None
    tcdf = df.loc[(slice(None), slice(None), ['dir', 'acc']), 'q90']#.reset_index(level=2, drop=True)
    
    #fig = plotcomp_tr_reg_dir_accs(tcdf, tcf, model, regressionmodel)
    fig = plotcomp_tr_reg_twovars(tcdf, ['dir', 'acc'], model, regressionmodel)
    fig.savefig(os.path.join(folder, 'tcctrlcomp_tr_reg_dir_accs_horlabels_shortlegend_new-function.pdf'))
    fig.savefig(os.path.join(folder, 'tcctrlcomp_tr_reg_dir_accs_horlabels_shortlegend_new-function.png'))
    fig.savefig(os.path.join(folder, 'tcctrlcomp_tr_reg_dir_accs_horlabels_shortlegend_new-function.svg'))

    print('made new dir acc plots')
    
    eedf = df.loc[(slice(None), slice(None), ['ee', 'eepolar']), 'q90']
    
    ee_fig = plotcomp_tr_reg_ees(eedf, model, regressionmodel)
    ee_fig.savefig(os.path.join(folder, 'tcctrlcomp_tr_reg_ees.pdf'))
    ee_fig.savefig(os.path.join(folder, 'tcctrlcomp_tr_reg_ees.svg'))
    
    comb_df = df.loc[(slice(None), slice(None), ['dir', 'ee']), 'q90']
    
    comb_fig = plotcomp_tr_reg_twovars(comb_df, ['dir', 'ee'], model, regressionmodel)
    comb_fig.savefig(os.path.join(folder, 'tcctrlcomp_tr_reg_comb.pdf'))
    comb_fig.savefig(os.path.join(folder, 'tcctrlcomp_tr_reg_comb.svg'))
    
    plt.close('all')
        
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
    fig.savefig(os.path.join(folder, 'tcctrlcomp_dir_accs_horlabels_shortlegend.svg'))
    
    eedf = df.loc[(slice(None), slice(None), ['ee', 'eepolar']), 'q90']
    
    ee_fig = plotcomp_ees(eedf, model)
    ee_fig.savefig(os.path.join(folder, 'tcctrlcomp_ees.pdf'))
    ee_fig.savefig(os.path.join(folder, 'tcctrlcomp_ees.svg'))
    
    comb_df = df.loc[(slice(None), slice(None), ['dir', 'ee']), 'q90']
    
    comb_fig = plotcomp_twovars(comb_df, ['dir', 'ee'], model)
    comb_fig.savefig(os.path.join(folder, 'tcctrlcomp_comb.pdf'))
    comb_fig.savefig(os.path.join(folder, 'tcctrlcomp_comb.svg'))
    
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

    linewidth=5.1
    
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
        #plt.plot(layers, np.nanmean(np.abs(alltmdevmeans[i]), axis=1), color=trainedmodel['color'], marker = 'D', alpha = 0.15, label='ind trained')
        plt.plot(layers, np.nanmean(np.abs(alltmdevmeans[i]), axis=1), color=trainedmodel['color'], marker = 'D', alpha = 0.15, label='ind trained', linewidth=linewidth)
        #plt.plot(layers, np.nanmean(np.abs(allcmdevmeans[i]), axis=1), color='grey', marker = 'D', alpha = 0.15, label='ind control')
        plt.plot(layers, np.nanmean(np.abs(allcmdevmeans[i]), axis=1), color='grey', marker = 'D', alpha = 0.15, label='ind untrained', linewidth=linewidth) ##eLife
        
    plt.errorbar(layers, tmsmean, yerr=errs_tmsmean, marker='D', color=trainedmodel['color'], capsize=3.0, label='mean of trained', linewidth=linewidth)
    plt.errorbar(layers, cmsmean, yerr=errs_cmsmean, marker = 'D', color='grey', capsize=3.0, label='mean of controls', linewidth=linewidth)

    plt.xticks(list(range(len(layers))), ['Sp.'] + ['L%d' %(i+1) for i in range(len(layers) - 1)])
    plt.xlim((-0.3, len(layers)-0.7))
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Deviation')
    
    ax = plt.gca()
    format_axis(ax)
    handles, _ = ax.get_legend_handles_labels()
    handles = np.array(handles)
    
    #plt.legend(handles[[0,1,10,11]], ['ind trained', 'ind control', \
    #            'mean of trained', 'mean of controls'])

    plt.legend(handles[[0,1,10,11]], ['Ind. Trained', 'Ind Untrained', \
               'Mean Trained', 'Mean Untrained']) ##eLife


    ## eLife Style
    ## SET AXIS WIDTHS
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(5.1)

    # increase tick width
    ax.tick_params(width=5.1)
    
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
    
    figboth_hor.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_hor_both_02.pdf'), dpi=300, transparent=True)
    figboth_hor.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_hor_both_02.svg'), dpi=300, transparent=True)
    figboth_hor.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_hor_both_02.png'), dpi=300, transparent=True)
    figboth_vert.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_vert_both_02.pdf'), dpi=300, transparent=True)
    figboth_vert.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_vert_both_02.svg'), dpi=300, transparent=True)
    figboth_vert.savefig(os.path.join(analysisfolder, modelbase + '_mean_all_ind_neuron_dev_mad_plot_vert_both_02.png'), dpi=300, transparent=True)
    
    df_hor.to_csv(os.path.join(analysisfolder, modelbase + '_deviations_mad_sig_hor.csv'))
    df_vert.to_csv(os.path.join(analysisfolder, modelbase + '_deviations_mad_sig_vert.csv'))
        
    print('plots saved')

# %% MAIN

def main(model, runinfo):
    """main for comparing features in a single plane for all models"""
    
    print('comparing kinetic differences for model %s ...' %model['base'])
    df = None
    decoding_df = None
    
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'kindiffs'))):
    #if(True):
    #if(runinfo.default_run):
    if(runinfo.default_run and runinfo['height'] == 'all'):
    #if(True and runinfo['height'] == 'all'):
        print('compiling dataframe for comparions...')
        df = compile_comparisons_df(model, runinfo)
        
    else:
        print('kinetic and label embeddings already analyzed')
    
    if(runinfo.default_run):
    #if(runinfo['height'] == 'all'):
        print('compiling dataframe for decoding comparions...')
        for alpha in alphas:
            decoding_df = compile_decoding_comparisons_df(model, runinfo, alpha)
        
    else:
        print('decoding comparisons already created already analyzed')

    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'pairedt'))):
    #if(True):
    if(runinfo.default_run):    
        print('running pairedt...')
        pairedt_comp(model, runinfo)
        print('ks analysis saved')
        
    else:
        print('kinetic and label embeddings already analyzed')
        
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'kindiffs_plots'))):
    #if(True):
    #if(runinfo.default_run):
    if(runinfo.default_run and runinfo['height'] == 'all'):
        print('running tcctrlcompplots for model %s ' %model['name'])
        if df is None:
            analysisfolder = runinfo.sharedanalysisfolder(model, 'kindiffs')
            df = pd.read_csv(os.path.join(analysisfolder, model['base'] + '_comparisons_df.csv'),
                             header=0, index_col=[0,1,2], dtype={'layer': int, 'mean': float, 'median': float})
        print('creating kindiffs plots')
        tcctrlcompplots(df, model, runinfo)
    else:
        print('kindiffs plots already made')

    #decoding kindiffs plots
    #if(runinfo.default_run):
    #if(runinfo.default_run and runinfo['height'] == 'all'):
    if(runinfo['height'] == 'all'):
        for alpha in alphas:
            '''
            if decoding_df is None:W
                analysisfolder = runinfo.sharedanalysisfolder(model, 'decoding_kindiffs')
                #SWITCH FOR NORMALIZATION
                #decoding_df = pd.read_csv(os.path.join(analysisfolder, model['base'] + '_decoding_comparisons_df_normalized.csv'),
                decoding_df = pd.read_csv(os.path.join(analysisfolder, model['base'] + '_decoding_comparisons_df.csv'),
                                header=0, index_col=[0,1,2], dtype={'layer': int, 'mean': float, 'median': float})
            '''
            decoding_df = compile_decoding_comparisons_df(model, runinfo, alpha)
            print('creating decoding kindiffs plots')
            decoding_tcctrlcompplots(decoding_df, model, runinfo, alpha)
    else:
        print('decoding kindiffs plots already made')

    if(runinfo.default_run):    
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'pd_deviation'))):
    #if(False):
    #if(True):
        print('computing deviation measure for PDs')
        pd_deviation(model, runinfo)
        print('df saved')
    else:
        print('pd deviation measure already saved')

def comparisons_tr_reg_main(taskmodel, regressionmodel, runinfo, alpha=None):
    
    print('comparing kinetic differences for model %s trained & reg ...' %taskmodel['base'])
    df = None
    decoding_df = None
    
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'kindiffs'))):
    #if(runinfo['height'] == 'all'):
    #if(runinfo.default_run):
    if(False):
        print('compiling dataframe for comparions trained & reg...')
        df = compile_comparisons_tr_reg_df(taskmodel, regressionmodel, runinfo)
        
    else:
        print('kinetic and label embeddings already analyzed')
    
    #if(runinfo.default_run):
    if(runinfo['height'] == 'all'):
    #if(False):
        for alpha in alphas:
            print('compiling dataframe for decoding comparions trained & reg...')
            decoding_df = compile_decoding_comparisons_tr_reg_df(taskmodel, regressionmodel, runinfo, alpha)
        
    else:
        print('decoding comparisons already created already analyzed')
        #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'pairedt'))):

    #if(True):
    if(runinfo['height'] == 'all'):
    #if(runinfo.default_run):    
        print('running pairedt...')
        pairedt_comp(taskmodel, runinfo, regressionmodel)
        print('ks analysis saved')
        
    else:
        print('kinetic and label embeddings already analyzed')

    if(runinfo['height'] == 'all'):
    #if(False):
        if df is None:
            analysisfolder = runinfo.sharedanalysisfolder(taskmodel, 'kindiffs_tr_reg')
            df = pd.read_csv(os.path.join(analysisfolder, taskmodel['base'] + '_comparisons_reg_tr_df.csv'),
                             header=0, index_col=[0,1,2], dtype={'layer': int, 'mean': float, 'median': float})

        print('running paired t quantiles')
        pairedt_quantiles(df, taskmodel, runinfo, regressionmodel)
        print('done')
    else:
        print('skipping pairedt quantiles')

    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'kindiffs_plots'))):
    if(runinfo['height'] == 'all' and runinfo.default_run):
    #if(runinfo['height'] == 'all' and True):
    #if(runinfo.default_run):
        if df is None:
            analysisfolder = runinfo.sharedanalysisfolder(taskmodel, 'kindiffs_tr_reg')
            df = pd.read_csv(os.path.join(analysisfolder, taskmodel['base'] + '_comparisons_reg_tr_df.csv'),
                             header=0, index_col=[0,1,2], dtype={'layer': int, 'mean': float, 'median': float})
        print('creating kindiffs plots trained & reg')
        tcctrlcompplots_tr_reg(df, taskmodel, regressionmodel, runinfo)
        print('should have reached them')
    else:
        print('kindiffs plots already made')

    #decoding kindiffs plots
    #if(runinfo.default_run):
    #if(False):
    if(runinfo['height'] == 'all'):
        for alpha in alphas:
            #if decoding_df is None:
            decoding_df = compile_decoding_comparisons_tr_reg_df(taskmodel, regressionmodel, runinfo, alpha)
            analysisfolder = runinfo.sharedanalysisfolder(taskmodel, 'decoding_kindiffs')
            #SWITCH FOR NORMALIZATION
            #decoding_df = pd.read_csv(os.path.join(analysisfolder, taskmodel['base'] + '_decoding_comparisons_tr_reg_df_normalized.csv'),
            #decoding_df = pd.read_csv(os.path.join(analysisfolder, taskmodel['base'] + '_decoding_comparisons_df.csv'),
            #                header=0, index_col=[0,1,2], dtype={'layer': int, 'mean': float, 'median': float})
            print(decoding_df)
            print('creating decoding kindiffs plots trained & reg')
            decoding_tcctrlcompplots_tr_reg(decoding_df, taskmodel, regressionmodel, runinfo, alpha)
    else:
        print('decoding kindiffs plots already made')

def generalizations_comparisons_main(model, runinfo):
    """main for comparing features that describe all planes"""
    
    #if(runinfo.default_run):
    #if(not os.path.exists(runinfo.sharedanalysisfolder(model, 'ind_neuron_invars_comp', False))):
    if(True):
        print('running individual neuron invars comparison...')
        ind_neuron_invars_comp(model, runinfo)
        print('saved plots')
    else:
        print('ind neuron invars comps already run')
        
    plt.close('all')

# %%
