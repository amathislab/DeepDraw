#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:06:05 2019

@author: kai
"""

import argparse
import os
import numpy as np
import yaml
import io
import copy
import matplotlib
from utils import sorted_alphanumeric

try:
    from savelouts_best import main as modeloutputs_main
    from regression_task_cka import main as regression_task_cka_main
except ModuleNotFoundError as e:
    print(e)
    print('proceeding without savelouts , this will only work if no data is being generated')

from rowwise_neuron_curves import main as tuningcurves_main
from combined_violinquantiles import comp_violin_main, comp_tr_reg_violin_main, comp_tr_reg_violin_main_newplots
from comparisons import main as comparisons_main
from comparisons import comparisons_tr_reg_main
from comparisons import generalizations_comparisons_main
from prefdir_controls import main as prefdir_main
from generalization import main as generalization_main
from representational_similarity_analysis import main as rsa_main
from representational_similarity_analysis import rsa_models_comp
from polar_tcs import main as polar_tcs_main
from tsne import main as tsne_main
from network_dissection import main as network_dissection_main
from unit_classification import main as unit_classification_main

def format_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

# %% CONFIGURATION OPTIONS

# #specify location in which model weights and results are to be saved
basefolder = '/media/data/DeepDraw/revisions/analysis-data/' #end on analysis-data/

# %% UTILS, CONFIG MODELS, AND GLOBAL VARS

fsets = ['vel', 'acc', 'labels', 'ee', 'eepolar',]
decoding_fsets = ['ee', 'eepolar', 'vel', 'acc']
orientations = ['hor', 'vert']
uniquezs = list(np.array([-45., -42., -39., -36., -33., -30., -27., -24., -21., -18., -15.,
                     -12.,  -9.,  -6.,  -3.,   0.,   3.,   6.,   9.,  12.,  15.,  18.,
                     21.,  24.,  27.,  30.]).astype(int))
uniquexs = list(np.array([ 6.,  9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42.,
                     45., 48., 51., 54.]).astype(int))
uniqueplanes=[uniquezs, uniquexs]

class RunInfo(dict):
    ''' This class is an extension of the dictionary storing info about the current experimental run.
    It includes methods that return formatted strings specifying the various folders in which the analysis is saved.
    '''

    def __init__(self, *args, **kwargs):
        super(RunInfo, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def get_exp_id(self, model):
        ''' Read out experimental ID from either model dict or runinfo config (based on that order of priority )
        
        Arguments
        ---------
        model : dict, contains model config info

        Returns
        -------
        exp_id : int, experimental ID tied to the analysis
        '''
        if model['exp_id'] is not None:
            return model['exp_id']
        else:
            return self.__dict__['expid']

    def planestring(self):
        ''' format information about which plane it is as a single string '''
        return self.__dict__['orientation'] + str(self.__dict__['height'])

    def experimentfolder(self):
        ''' return base folder corresponding to this experimental ID '''
        return '%sexp%d' %(self.__dict__['basefolder'], self.__dict__['expid'])

    def resultsfolder(self, model, fset = None):
        ''' Returns the formatting string specifying the folder name in which the tuning curve fit test scores are saved.

        Arguments
        ---------
        model : dict, information about the model run
        fset : string, tuning curve type

        Returns
        -------
        resultsfolder : string, folder in which tuning curve fit test scores are saved
        '''

        resultsfolder =  '%sexp%d/results/%s/%s/%s/' %(self.__dict__['basefolder'], self.__dict__['expid'], model['base'], model['name'], self.planestring())
        if fset is not None:
            resultsfolder = os.path.join(resultsfolder, fset)
        return resultsfolder

    def sharedanalysisfolder(self, model, analysis = None, sp=True):
        ''' Returns the formatting string specifying the folder name in which analysis are stored that are about
        all five model instantiations within a given type

        Arguments
        ---------
        model : dict, information about the model run
        analysis : string, name of analysis that is being run
        sp : bool, are we computing the results for a single plane

        Returns
        -------
        sharedanalysisfolder : string, folder name in which analyses are stored that are about all five model
        instantiations within a given type
        '''

        expid = self.get_exp_id(model)

        if sp:
            sharedanalysisfolder = '%sexp%d/analysis/%s/comparison/%s/' %(self.__dict__['basefolder'], expid, model['base'], self.planestring())
        else:
            sharedanalysisfolder = '%sexp%d/analysis/%s/comparison/' %(self.__dict__['basefolder'], expid, model['base'])
        if analysis is not None:
            sharedanalysisfolder = os.path.join(sharedanalysisfolder, analysis)
        return sharedanalysisfolder

    def baseanalysisfolder(self, model, analysis=None):
        ''' Helper function that returns the main analysis folder for an individual model instantiation

        Arguments
        ---------
        model : dict, information about the model run
        analysis : string, name of analysis that is being run

        Returns
        -------
        baseanalysisfolder : string, main analysis folder for an individual model instantiation
        '''

        expid = self.get_exp_id(model)

        baseanalysisfolder = '%sexp%d/analysis/%s/%s/' %(self.__dict__['basefolder'], expid, model['base'], model['name'])
        if analysis is not None:
            baseanalysisfolder = os.path.join(baseanalysisfolder, analysis)
        return baseanalysisfolder

    def analysisfolder(self, model, analysis=None):
        ''' Returns the formatting string specifying the folder name in which the analysis is stored

        Arguments
        ---------
        model : dict, information about the model run
        analysis : string, name of analysis that is being run

        Returns
        -------
        analysisfolder : string, folder in which analysis is stored
        '''

        baseanalysisfolder = self.baseanalysisfolder(model)
        analysisfolder = os.path.join(self.__dict__['basefolder'], baseanalysisfolder, self.planestring())
        if analysis is not None:
            analysisfolder = os.path.join(analysisfolder, analysis)
        return analysisfolder

    def generalizationfolder(self, model, analysis=None):
        ''' Returns the formatting string specifying the folder name in which an analysis of generalizational capacity can be stored

        Arguments
        ---------
        model : dict, information about the model run
        analysis : string, name of analysis that is being run

        Returns
        -------
        generalizationfolder : string
        '''

        generalizationfolder = os.path.join(self.__dict__['basefolder'], self.baseanalysisfolder(model), 'generalization')
        if analysis is not None:
            generalizationfolder = os.path.join(generalizationfolder, analysis)
        return generalizationfolder

    def datafolder(self, model = None):
        ''' Returns the formatted string specifying the folder name in which the hidden layer activations are stored

        Arguments
        ---------
        model : dict, information about the model run

        Returns
        -------
        datafolder : string
        '''
    
        expid = self.get_exp_id(model)

        datafolder = '%sexp%d/data/' %(self.__dict__['basefolder'], expid)
        if model is not None:
            datafolder = os.path.join(datafolder , model['base'], model['name'])
        return datafolder

    def allmodelfolder(self, analysis = None, sp=True):
        ''' Returns the formatted string specifying the folder name in which analyses are stored that compare several model types

        Arguments
        ---------
        analysis : string, name of analysis that is being run
        sp : bool, are we computing the results for a single plane

        Returns
        -------
        datafolder : string
        '''

        allmodelfolder = '%sexp%d/analysis/combined/' %(self.__dict__['basefolder'], self.__dict__['expid'])
        if sp:
            allmodelfolder = os.path.join(allmodelfolder, self.planestring())
        if analysis is not None:
            allmodelfolder = os.path.join(allmodelfolder, analysis)
        return allmodelfolder

    def regressiontaskfolder(self, model, analysis = None):
        ''' Returns the formatted string specifying the folder name in which analyses are stored relating to regression task

        Arguments
        ---------
        analysis : string, name of analysis that is being run
        sp : bool, are we computing the results for a single plane

        Returns
        -------
        datafolder : string
        '''

        expid = self.get_exp_id(model)

        regressiontaskfolder = '%sexp%d/analysis/regression_task/' %(self.__dict__['basefolder'], expid)
        if analysis is not None:
            regressiontaskfolder = os.path.join(regressiontaskfolder, analysis)
        return regressiontaskfolder

# %% EXPERIMENTAL RUN CONFIG

runinfo = RunInfo({'expid': 402, #internal experiment id
                   'datafraction': 'auto',  #fraction (0,1] or 'auto' (i.e. if you want to run a new analysis but keep the old results that it would otherwise overwrite, increment by 1)
                   'randomseed': 2000,
                   'randomseed_traintest': 42,
                   'dirr2threshold': 0.2,
                   'verbose': 2, #0 (least), 1, 2 (most)
                   'model_experiment_id': 'auto',  #used in model training, int or 'auto'
                   'basefolder': basefolder,
                   'batchsize': 100, #for layer representation generation
                   'default_run': True, #only variable that is 'trial'-dependent,
                                    #ie should be changed when rerunning stuff in same folder
                                    #not semantically important for run info
                    'dpi': 500
            })

exp_par_lookup = {
    402: {'datafraction': 0.1, 'model_experiment_id': 22 }, # S - TDT
    403: {'datafraction': 0.1, 'model_experiment_id': 22 }, # ST - TDT
    405: {'datafraction': 0.05, 'model_experiment_id': 32 }, # LSTM - TDT
    406: {'datafraction': 0.1, 'model_experiment_id': 23 }, # S - ART
    407: {'datafraction': 0.1, 'model_experiment_id': 23 }, # ST - ART
    408: {'datafraction': 0.05, 'model_experiment_id': 33 }, # LSTM - ART
}

# %% SAVE OUTPUTS AND RUN ANALYSIS

def main(do_data=False, do_results=False, do_analysis=False, do_regression_task = False, include = ['S', 'T','ST'], tasks = ['task'], expid = None, reg_exp_id = None):
    ''' Calls the analyses that need to be run for all model types and instantiations

    Parameters
    ----------
    do_data : bool, should hidden layer activations be extracted
    do_results : bool, should the tuning curves be fitted
    do_analysis : bool, should we fit the analysis strengths
    include : list of strings, short names of different model types for which the functions are supposed to runs
    '''

    if expid is not None:
        runinfo['expid'] = expid
        if runinfo['datafraction'] == 'auto':
            runinfo['datafraction'] = exp_par_lookup[expid]['datafraction']

    if runinfo['model_experiment_id'] == 'auto':
        print(expid, exp_par_lookup[expid])
        runinfo['model_experiment_id'] = exp_par_lookup[expid]['model_experiment_id']

    print("Running Experiment %d with datafraction %.2f" %(expid, runinfo['datafraction']))

    matplotlib.pyplot.rcParams.update({'legend.title_fontsize':40})

    print('experimental id %d' %runinfo['expid'])
    print('data fraction %.2f' %runinfo['datafraction'])

    os.makedirs(runinfo.experimentfolder(), exist_ok=True)
    configfilename = os.path.join(runinfo.experimentfolder(), 'config.yaml')
    with io.open(configfilename, 'w',  encoding='utf8') as outfile:
        yaml.dump(runinfo, outfile, default_flow_style=False, allow_unicode=True)
        
    allmodels = [
        dict({'type': 'S',
            'typename': 'spatial_temporal',
            'base': 'spatial_temporal_4_8-16-16-32_32-32-64-64_7293',
            'base_regression': 'spatial_temporal_r_4_8-16-16-32_32-32-64-64_7293',
            'nlayers': 8,
            'max_nlayers': 8,
            'max_act': 14, #this can be manually adjusted as the maximum in the preferred direction histogram
            'control': False,
            'cmap': matplotlib.colors.ListedColormap(['midnightblue']),
            'color': 'midnightblue',
            'regression_color': 'darkturquoise',
            'control_cmap': 'Greys_r',
            'regression_cmap': matplotlib.colors.ListedColormap(['darkturquoise']),
            's_stride': 2,
            't_stride': 3,
            'regression_task': False,
            'model_path': None,
            'exp_id': None,}),
        dict({'type': 'ST',
              'typename': 'spatiotemporal',
              'base': 'spatiotemporal_4_8-8-32-64_7272',
              'base_regression': 'spatiotemporal_r_4_8-8-32-64_7272',
              'nlayers': 4,
              'max_nlayers': 4,
              'max_act': 14, #this can be manually adjusted as the maximum in the preferred direction histogram
              'control': False,
              'cmap': matplotlib.colors.ListedColormap(['darkgreen']),
              'color': 'darkgreen',
              'regression_color': 'yellowgreen',
              'control_cmap': 'Greys_r',
              'regression_cmap': matplotlib.colors.ListedColormap(['yellowgreen']),    
              't_stride': 2,
              's_stride': 2,
              'regression_task': False,
              'model_path': None,
              'exp_id': None,}),
        dict({'type': 'LSTM',
            'typename': 'recurrent',
            'base': 'lstm_3_8-16-16_256',
            'base_regression': 'lstm_r_3_8-16-16_256',
            'nlayers': 4,
            'max_nlayers': 5,
            'max_act': 14, #this can be manually adjusted as the maximum in the preferred direction histogram
            't_stride': 1,
            's_stride': 1,
            'control': False,
            'cmap': matplotlib.colors.ListedColormap(['indigo']),
            'color': 'indigo',
            'regression_color': 'orchid',
            'control_cmap': 'Greys_r',
            'regression_cmap': matplotlib.colors.ListedColormap(['orchid']),
            'regression_task': False,
            'model_path': None,
            'exp_id': None,
            })
        ]

    models = [model for model in allmodels if (model['type'] in include)]

    runinfo.regression_task = do_regression_task
        #slightly confusing where runinfo.regression_task specifies if we are running CKA regression cluster of tasks,
        #whereas modelinfo.regression_task specifies if the current model was trained on regression (True) or task (False)

    print("beginning body...")

    startmodel = 0
    startrun = 1 ### 0 for full run
    startcontrol = False
    startior = 0
    startheight = 'all'

    endrun = 6
    endheight = None

    ## Utility functions for late start
    runmodels = False
    runruns = False
    runtype = False
    runior = False
    runheight = False

    default_run = runinfo.default_run
    initiated_mp = False

    for imodel, model in enumerate(models):
        
        #print(startmodel, imodel)

        if not do_regression_task:

            for task in tasks:

                if task == 'regression':
                    print("Working on regression models...")
                    model['base'] = model['base_regression']
                    model['regression_task'] = True
                    model['color'] = model['regression_color']
                    model['cmap'] = model['regression_cmap']

                    if reg_exp_id is not None:
                        model['exp_id'] = reg_exp_id
                
                if startmodel == imodel:
                    print("Running model ", imodel)
                    runmodels = True

                if runmodels:
                    for i in np.arange(1, 6):

                        modelname = model['base'] + '_%d' %i
                        model['name'] = modelname

                        model_to_analyse = model.copy()
                        trainedmodel = model.copy()
                        
                        regressionmodel = model.copy()
                        regressionmodel['base'] = model['base_regression']
                        regressionmodel['name'] = regressionmodel['base'] + '_%d' %i
                        regressionmodel['regression_task'] = True

                        if startrun == i:
                            print("Running model run ", startrun)
                            runruns = True

                        if endrun < i:
                            print("Ending on model run %d since endrun surpassed" %endrun)
                            runruns = False

                        if runruns:
                            for control in [False, True]:

                                if control:
                                    modelname = modelname + 'r'
                                    model_to_analyse['name'] = modelname
                                    model_to_analyse['control'] = True

                                if startcontrol == control:
                                    runtype = True

                                if runtype:
                                    if(do_data):
                                        if(True):
                                        #if(default_run):
                                            print('generating output for model %s ...' %modelname)
                                            modeloutputs_main(model_to_analyse, runinfo)
                                        else:
                                            print('data for model %s already generated' %modelname)
                                    
                                    if(do_results or do_analysis):
                                        
                                        for ior, orientation in enumerate(orientations):
                                            runinfo['orientation'] = orientation

                                            if startior == ior:
                                                runior = True

                                            if runior:
                                            #if(False):
                                                for height in (['all'] + uniqueplanes[ior]):
                                                    runinfo['height'] = height

                                                    runinfo_to_analyse = copy.copy((runinfo))

                                                    if startheight == height:
                                                        runheight = True

                                                    if endheight is not None and endheight == 'after_all' and height != 'all':
                                                        runheight = False
                                                        print("Ending run since finished run for plane all")

                                                    if runheight:

                                                        if(do_results):
                                                            print('running analysis for model %s plane %s...' %(modelname, runinfo.planestring()))

                                                            for fset in fsets:

                                                                #if(not os.path.exists(runinfo.resultsfolder(model_to_analyse, fset))):
                                                                if(default_run and height == 'all'):
                                                                #if(True):
                                                                #if(False):
                                                                
                                                                    print('running %s analysis (fitting tuning curves) for model %s plane %s...' %(fset, modelname, runinfo.planestring()))
                                                                    tuningcurves_main(fset,
                                                                                    runinfo_to_analyse,
                                                                                    model_to_analyse,
                                                                                    #pool=pool
                                                                                    )

                                                                else:
                                                                    print('%s analysis for model %s plane %s already completed' %(fset, modelname, runinfo.planestring()))                                                  

                                                            for dfset in decoding_fsets:
                                                                #if(default_run and height == 'all'): 
                                                                if(True and height == 'all'): 
                                                                    #for alpha in [0, 0.001, 0.01, 0.1, 1.0, 5.0, 10, 100, 1000, 10000, 100000, 1000000]:
                                                                    #for alpha in [0.01, 1, 100, 10000]:
                                                                    for alpha in [1]:
                                                                        print('decoding %s analysis for model %s plane %s with regularization par %f...' %(fset, modelname, runinfo.planestring(), alpha))
                                                                        tuningcurves_main(dfset,
                                                                                        runinfo_to_analyse,
                                                                                        model_to_analyse,
                                                                                        mmod='decoding',
                                                                                        alpha=alpha
                                                                                        )
                                                                else:
                                                                    print('decoding %s analysis for model %s plane %s already completed' %(fset, modelname, runinfo.planestring()))                                                  

                                                        if(do_analysis):
                                                            #check to make sure that this model and plane combination has any samples
                                                            try:
                                                                has_samples = ( len(np.load(os.path.join(runinfo_to_analyse.resultsfolder(model_to_analyse, 'vel'), 'l%d_%s_mets_%s_%s_test.npy' %(0, 'vel', 'std', runinfo_to_analyse.planestring())))) > 0 )
                                                            except FileNotFoundError:
                                                                try:
                                                                    has_samples = ( len(np.load(os.path.join(runinfo_to_analyse.resultsfolder(model_to_analyse, 'decoding_vel'), 'l%d_%s_mets_%s_%s_a0_test.npy' %(0, 'vel', 'decoding', runinfo_to_analyse.planestring())))) > 0 )
                                                                except FileNotFoundError as e:
                                                                    print("No files found for this plane", e)
                                                            
                                                            if(has_samples):
                                                            #if(True):
                                                                print('compiling results and generating graphs for model %s plane %s...' %(modelname, runinfo.planestring()))

                                                                print('generating polar tuning curve plots for model %s plane %s ...' %(modelname, runinfo.planestring()))
                                                                
                                                                #if(not os.path.exists(runinfo.analysisfolder(model_to_analyse, 'polar_tcs'))):
                                                                #if(True):
                                                                #if(default_run):
                                                                #if(False):
                                                                if(runinfo.planestring() == 'horall'):
                                                                    polar_tcs_main(model_to_analyse, runinfo_to_analyse)
                                                                else:
                                                                    print('polar tc plots already exist')
                                                                
                                                                print('generating preferred direction histograms for model %s plane %s...' %(modelname, runinfo.planestring()))
                                                                #if(not os.path.exists(runinfo.analysisfolder(model_to_analyse, 'prefdir'))):
                                                                #if(False):
                                                                if(default_run and runinfo.planestring() == 'horall'):

                                                                    prefdir_main(model_to_analyse, runinfo_to_analyse)
                                                                else:
                                                                    print('pref dir plots already exist')
                                                                    
                                                                #if(not os.path.exists(runinfo.analysisfolder(trainedmodel, 'tsne'))):
                                                                #if(True):
                                                                if(default_run):
                                                                #if(False):
                                                                    print('plotting tSNE for model %s plane %s .... ' %(modelname, runinfo.planestring()))
                                                                    tsne_main(model_to_analyse, runinfo_to_analyse)

                                                                #if(default_run):
                                                                #if(False):
                                                                if(True and runinfo.planestring() == 'horall'):
                                                                    print('running unit classification...')
                                                                    unit_classification_main(model_to_analyse, runinfo)
                                                                else:
                                                                    print('unit classification already exists')

                                                                if(i == 1 and runinfo.planestring() == 'horall' and not control):
                                                                    if(False):
                                                                        print('running network dissection for model %s plane %s .... ' %(modelname, runinfo.planestring()))
                                                                        network_dissection_main(model_to_analyse, runinfo_to_analyse)
                                                                    else:
                                                                        print('network dissection already completed or skipped')

                                                                if(control):
                                                                    #if(not os.path.exists(runinfo.analysisfolder(trainedmodel, 'comp_violin'))):
                                                                    #if(default_run):
                                                                    #if(True):
                                                                    #if(False):
                                                                    #if(default_run and runinfo.planestring() == 'horall'):
                                                                    if(False and runinfo.planestring() == 'horall'):
                                                                        print('saving comparison violin plot for model %s plane %s...' %(modelname, runinfo.planestring()))
                                                                        comp_violin_main(trainedmodel, model_to_analyse, runinfo)
                                                                    else:
                                                                        print('comparison violin plot already saved')

                                                                    if(runinfo.planestring() == 'horall'):
                                                                        #if(not os.path.exists(runinfo.analysisfolder(trainedmodel, 'rsa'))):
                                                                        if(True):
                                                                        #if(default_run):
                                                                        #if(False):
                                                                            print('computing representational similarity analysis for model %s plane %s ... ' %(modelname, runinfo.planestring()))
                                                                            rsa_main(trainedmodel, model_to_analyse, runinfo)

                                                                        else:
                                                                            print('rsa already saved')
                                                                    
                                                                else:

                                                                    #check to make sure that this model and plane combination has any samples
                                                                    # #if(len(np.load(os.path.join(runinfo_to_analyse.resultsfolder(regressionmodel, 'vel'), 'l%d_%s_mets_%s_%s_test.npy' %(0, 'vel', 'std', runinfo_to_analyse.planestring())))) > 0):
                                                                    if(True): ### NEW PLOT TYPE BELOW
                                                                    #if(False):
                                                                
                                                                        #if(False):  
                                                                        #if(True):  
                                                                        if(runinfo.planestring() == 'horall'):
                                                                            print("saving violin plot comparison reg & task-trained for model %s plane %s ... " %(modelname, runinfo.planestring()))
                                                                            comp_tr_reg_violin_main(model_to_analyse, regressionmodel, runinfo)

                                                                    if(True):
                                                                    #if(len(np.load(os.path.join(runinfo_to_analyse.resultsfolder(regressionmodel, 'vel'), 'l%d_%s_mets_%s_%s_test.npy' %(0, 'vel', 'std', runinfo_to_analyse.planestring())))) > 0):
                                                                    #if(False):        
                                                                        if(True):
                                                                        #if(default_run and runinfo.planestring() == 'horall'):
                                                                        #if(False):
                                                                            print("doing trreg cka for model %s plane %s ... " %(modelname, runinfo.planestring()))
                                                                            rsa_main(model_to_analyse, regressionmodel, runinfo, trreg=True)

                                                                        if(default_run and runinfo.planestring() == 'horall'):
                                                                        #if(True and runinfo.planestring() == 'horall'):
                                                                            print("making new violin plots")
                                                                            comp_tr_reg_violin_main_newplots(trainedmodel, regressionmodel, runinfo)

                                                                if (i==5):
                                                                    if(control):
                                                                        #if(False):
                                                                        if(default_run and runinfo.planestring() == 'horall'):
                                                                        #if(True and runinfo.planestring() == 'horall'):

                                                                            comparisons_main(model, runinfo)
                                                                        else:
                                                                            print('skipping comparisons')

                                                                        

                                                                        if(runinfo.planestring() == 'horall'):
                                                                        #if(True):
                                                                            print('combining rsa results for all models')
                                                                            rsa_models_comp(model, runinfo)
                                                                        else:
                                                                            print('rsa models comp already completed')
                                                                    else:
                                                                        if(task == 'task' and runinfo.planestring() == 'horall'):
                                                                            #if(True):
                                                                            #if(False):
                                                                            if(default_run):
                                                                                print('starting comparisons_tr_reg_main')
                                                                                comparisons_tr_reg_main(trainedmodel, regressionmodel, runinfo)


                                            else:
                                                runheight = True

                                            if(do_analysis):

                                                if startheight == 'comp':
                                                    runheight = True

                                                if runheight:
                                                    #if(False):
                                                    #if(True):
                                                    if(default_run):
                                                        print('launching analysis of nodes\' generalizational capacity...')
                                                        generalization_main(model_to_analyse, runinfo)

                                                    if(i==5):
                                                        if(control):
                                                            #if(True):
                                                            #if(False):
                                                            if(default_run):
                                                                generalizations_comparisons_main(model, runinfo)
                                                        else:
                                                            pass

        else:
            print('analyzing models on specified regression task')

            base_model_path = os.path.join(runinfo.basefolder, 'models', 'ALL_%s' %model['typename'])
            print(base_model_path)

            #print(next(os.walk(base_model_path))[1])
            all_regression_models = [f.name for f in os.scandir(base_model_path) if f.is_dir()]
            all_regression_models = sorted_alphanumeric(all_regression_models)
            print('all regression models: ', all_regression_models)
            task_models = [x for x in all_regression_models if ('_r_' not in x) ]
            regression_models = [x for x in all_regression_models  if ('_r_' in x) ]

            #all_regression_models_zipped = zip(task_models, regression_models)

            assert do_data and do_analysis, "data and analysis computed in one go for regression, need to have both activated"
            regression_task_cka_main(task_models, regression_models, runinfo, model.copy())

    print("Yay! Done, success")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Which parts of analysis to complete.')
    parser.add_argument('--data', type=bool, default = False, help='Extract data?')
    parser.add_argument('--results', type=bool, default=False, help='Fit TCs?')
    parser.add_argument('--analysis', type=bool, default=False, help='Analyze fitted TCs?')
    parser.add_argument('--regression_task', type=bool, default=False, help='Analyze models from regression task?')
    parser.add_argument('--task_models', type=bool, default=False, help='Include task models?')
    parser.add_argument('--regression_models', type=bool, default=False, help='Include regression models?')
    parser.add_argument('--S', type=bool, default=False, help='Include Spatial_temporal models?')
    parser.add_argument('--ST', type=bool, default=False, help='Include SpatioTemporal models?')
    parser.add_argument('--LSTM', type=bool, default=False, help='Include Spatial_temporal models?')
    parser.add_argument('--expid', type=int, default=None, help='What Experimental ID to use?')

    args = parser.parse_args()

    include = []
    if args.S:
        include.append('S')
    if args.ST:
        include.append('ST')
    if args.LSTM:
        include.append('LSTM')
    if (include == []):
        include = ['LSTM', 'S', 'T', 'ST']

    tasks = []
    if args.task_models:
        tasks.append('task')
    if args.regression_models:
        tasks.append('regression')

    print("Working on the following tasks: ", tasks)

    print(args.data, args.results, args.analysis)

    main(args.data, args.results, args.analysis, args.regression_task, include, tasks= tasks, expid= args.expid)
