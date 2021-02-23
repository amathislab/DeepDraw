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

try:
    from savelouts_best_controls import main as modeloutputs_main
except ModuleNotFoundError as e:
    print(e)
    print('proceeding without savelouts , this will only work if no data is being generated')

from rowwise_neuron_curves_controls import main as tuningcurves_main
from combined_violinquantiles_controls import comp_violin_main
from control_comparisons import main as comparisons_main
from control_comparisons import generalizations_comparisons_main
from prefdir_controls import main as prefdir_main
from generalization import main as generalization_main
from representational_similarity_analysis import main as rsa_main
from representational_similarity_analysis import rsa_models_comp
from polar_tcs import main as polar_tcs_main
from tsne import main as tsne_main
from network_dissection import main as network_dissection_main
from regression_task_cka import main as regression_task_cka_main

def format_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

# %% CONFIGURATION OPTIONS
basefolder = '/media/data/DeepDraw/revisions/analysis-data/' #end on analysis-data/

#basefolder = '/home/kai/Dropbox/DeepDrawData/analysis-data/' 

#specify location in which model weights and results are to be saved
                       # (trailing space)
    ## CHANGE THIS TO MATCH THE LOCATION OF THE 'analysis-data/' FOLDER ON DROPBOX

# %% UTILS, CONFIG MODELS, AND GLOBAL VARS

fsets = ['vel', 'acc', 'labels', 'ee', 'eepolar']
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

    def planestring(self):
        return self.__dict__['orientation'] + str(self.__dict__['height'])

    def experimentfolder(self):
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

        if sp:
            sharedanalysisfolder = '%sexp%d/analysis/%s/comparison/%s/' %(self.__dict__['basefolder'], self.__dict__['expid'], model['base'], self.planestring())
        else:
            sharedanalysisfolder = '%sexp%d/analysis/%s/comparison/' %(self.__dict__['basefolder'], self.__dict__['expid'], model['base'])
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

        baseanalysisfolder = '%sexp%d/analysis/%s/%s/' %(self.__dict__['basefolder'], self.__dict__['expid'], model['base'], model['name'])
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
        
        #print(self.__dict__['basefolder'])

        datafolder = '%sexp%d/data/' %(self.__dict__['basefolder'], self.__dict__['expid'])
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

        regressiontaskfolder = '%sexp%d/analysis/regression_task/' %(self.__dict__['basefolder'], self.__dict__['expid'])
        if analysis is not None:
            regressiontaskfolder = os.path.join(regressiontaskfolder, analysis)
        return regressiontaskfolder

# %% EXPERIMENTAL RUN CONFIG

runinfo = RunInfo({'expid': 308, #internal experiment id
                   'datafraction': 0.5,
                   #'datafraction': 0.5,
                   'randomseed': 2000,
                   'randomseed_traintest': 42,
                   'dirr2threshold': 0.2,
                   'verbose': 2, #0 (least), 1, 2 (most)
                   'model_experiment_id': 7, #as per Pranav's model generation
                   'basefolder': basefolder,
                   'batchsize': 100, #for layer representation generation
                   'default_run': False #only variable that is 'trial'-dependent,
                                    #ie should be changed when rerunning stuff in same folder
                                    #not semantically important for run info
            })

# %% SAVE OUTPUTS AND RUN ANALYSIS

def main(do_data=False, do_results=False, do_analysis=False, do_regression_task = False, include = ['S', 'T','ST']):
    ''' Calls the analyses that need to be run for all model types and instantiations

    Parameters
    ----------
    do_data : bool, should hidden layer activations be extracted
    do_results : bool, should the tuning curves be fitted
    do_analysis : bool, should we fit the analysis strengths
    include : list of strings, short names of different model types for which the functions are supposed to runs
    '''

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
            #'base': 'spatial_temporal_4_8-16-16-32_64-64-64-64_5272',
            'base': 'spatial_temporal_4_8-16-16-32_32-32-64-64_7293',
            'nlayers': 8,
            'max_act': 14, #this can be manually adjusted as the maximum in the preferred direction histogram
            'control': False,
            'cmap': 'Blues_r',
            'color': 'C0',
            'control_cmap': 'Purples_r',
            's_stride': 2,
            't_stride': 3}),
        dict({'type': 'ST',
              'typename': 'spatiotemporal',
              'base': 'spatiotemporal_4_8-8-32-64_7272',
              'nlayers': 4,
              'max_act': 14, #this can be manually adjusted as the maximum in the preferred direction histogram
              'control': False,
              'cmap': 'Greens_r',
              'color': 'green',
              'control_cmap': 'Greys_r',
              't_stride': 2,
              's_stride': 2}),
        dict({'type': 'LSTM',
            'typename': 'lstm',
            'base': 'lstm_3_8-16-16_256',
            'nlayers': 3,
            'max_act': 14, #this can be manually adjusted as the maximum in the preferred direction histogram
            't_stride': 1,
            's_stride': 1,
            'control': False,
            'cmap': 'Purples_r',
            'color': 'C4',
            'control_cmap': 'Purples_r'})
        ]

    models = [model for model in allmodels if (model['type'] in include)]

    print("beginning body...")

    startmodel = 0
    startrun = 1
    startcontrol = False
    startior = 0
    startheight = 'all'

    runmodels = False
    runruns = False
    runtype = False
    runior = False
    runheight = False

    default_run = runinfo.default_run

    for imodel, model in enumerate(models):
        
        #print(startmodel, imodel)

        if not do_regression_task:

            if startmodel == imodel:
                runmodels = True

            if runmodels:
                for i in np.arange(1, 6):

                    modelname = model['base'] + '_%d' %i
                    model['name'] = modelname

                    model_to_analyse = model.copy()
                    trainedmodel = model.copy()

                    if startrun == i:
                        runruns = True

                    if runruns == True:
                        for control in [False, True]:

                            if control:
                                modelname = modelname + 'r'
                                model_to_analyse['name'] = modelname
                                model_to_analyse['control'] = True

                            if startcontrol == control:
                                runtype = True

                            if runtype:
                                if(do_data):
                                    #if(not os.path.exists(runinfo.datafolder(model_to_analyse))):
                                    #if(True):
                                    if(default_run):
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
                                            for height in (['all'] + uniqueplanes[ior]):
                                                runinfo['height'] = height

                                                runinfo_to_analyse = copy.copy((runinfo))

                                                if startheight == height:
                                                    runheight = True

                                                if runheight:
                                                    if(do_results):
                                                        print('running analysis for model %s plane %s...' %(modelname, runinfo.planestring()))

                                                        for fset in fsets:

                                                            #if(not os.path.exists(runinfo.resultsfolder(model_to_analyse, fset))):
                                                            if(default_run):
                                                            #if(True):
                                                            
                                                                print('running %s analysis (fitting tuning curves) for model %s plane %s...' %(fset, modelname, runinfo.planestring()))
                                                                tuningcurves_main(fset,
                                                                                runinfo_to_analyse,
                                                                                model_to_analyse,
                                                                                )

                                                            else:
                                                                print('%s analysis for model %s plane %s already completed' %(fset, modelname, runinfo.planestring()))                                                  

                                                        for dfset in decoding_fsets:
                                                            #if(default_run):
                                                            if(True):
                                                                print('decoding %s analysis for model %s plane %s...' %(fset, modelname, runinfo.planestring()))
                                                                tuningcurves_main(dfset,
                                                                                runinfo_to_analyse,
                                                                                model_to_analyse,
                                                                                mmod='decoding'
                                                                                )
                                                            else:
                                                                print('decoding %s analysis for model %s plane %s already completed' %(fset, modelname, runinfo.planestring()))                                                  

                                                    if(do_analysis):
                                                        #evals = np.load(os.path.join(runinfo_to_analyse.resultsfolder(model_to_analyse, 'vel'), 'l%d_%s_mets_%s_%s_test.npy' %(0, 'vel', 'std', runinfo_to_analyse.planestring())))
                                                        #print(os.path.join(runinfo_to_analyse.resultsfolder(model_to_analyse, 'vel'), 'l%d_%s_mets_%s_%s_test.npy' %(0, 'vel', 'std', runinfo_to_analyse.planestring())))
                                                        #print(evals.shape)

                                                        if(len(np.load(os.path.join(runinfo_to_analyse.resultsfolder(model_to_analyse, 'vel'), 'l%d_%s_mets_%s_%s_test.npy' %(0, 'vel', 'std', runinfo_to_analyse.planestring())))) > 0):
                                                            print('compiling results and generating graphs for model %s plane %s...' %(modelname, runinfo.planestring()))

                                                            print('generating polar tuning curve plots for model %s plane %s ...' %(modelname, runinfo.planestring()))
                                                            
                                                            #if(not os.path.exists(runinfo.analysisfolder(model_to_analyse, 'polar_tcs'))):
                                                            #if(True):
                                                            if(default_run):
                                                                polar_tcs_main(model_to_analyse, runinfo_to_analyse)
                                                            else:
                                                                print('polar tc plots already exist')
                                                            
                                                            '''
                                                            try:
                                                                from subprocess32 import check_call
                                                                check_call(['test', '-f', runinfo.analysisfolder(model_to_analyse, 'polar_tcs')], timeout=0.5)
                                                            except:
                                                                print('did not find folder with subprocess')
                                                            else:
                                                                polar_tcs_main(model_to_analyse, runinfo_to_analyse)
                                                            '''

                                                            print('generating preferred direction histograms for model %s plane %s...' %(modelname, runinfo.planestring()))
                                                            if(not os.path.exists(runinfo.analysisfolder(model_to_analyse, 'prefdir'))):
                                                            #if(True):
                                                                prefdir_main(model_to_analyse, runinfo_to_analyse)
                                                            else:
                                                                print('pref dir plots already exist')
                                                                
                                                            #if(not os.path.exists(runinfo.analysisfolder(trainedmodel, 'tsne'))):
                                                            #if(True):
                                                            if(default_run):
                                                                print('plotting tSNE for model %s plane %s .... ' %(modelname, runinfo.planestring()))
                                                                tsne_main(model_to_analyse, runinfo_to_analyse)

                                                            if(i == 1 and runinfo.planestring() == 'horall' and not control):
                                                                if(True):
                                                                    print('running network dissection for model %s plane %s .... ' %(modelname, runinfo.planestring()))
                                                                    network_dissection_main(model_to_analyse, runinfo_to_analyse)
                                                                else:
                                                                    print('network dissection already completed or skipped')

                                                            if(control):
                                                                #if(not os.path.exists(runinfo.analysisfolder(trainedmodel, 'comp_violin'))):
                                                                if(default_run):
                                                                #if(True):
                                                                    print('saving comparison violin plot for model %s plane %s...' %(modelname, runinfo.planestring()))
                                                                    comp_violin_main(trainedmodel, model_to_analyse, runinfo)
                                                                else:
                                                                    print('comparison violin plot already saved')

                                                                if(runinfo.planestring() == 'horall'):
                                                                    #if(not os.path.exists(runinfo.analysisfolder(trainedmodel, 'rsa'))):
                                                                    #if(True):
                                                                    if(default_run):
                                                                        print('computing representational similarity analysis for model %s plane %s ... ' %(modelname, runinfo.planestring()))
                                                                        rsa_main(trainedmodel, model_to_analyse, runinfo)

                                                                    else:
                                                                        print('rsa already saved')

                                                            if (i==5 and control):
                                                                if(False):
                                                                    comparisons_main(model, runinfo)
                                                                else:
                                                                    print('skipping comparisons')

                                                                if(runinfo.planestring() == 'horall'):
                                                                    print('combining rsa results for all models')
                                                                    #if(not os.path.exists(runinfo.sharedanalysisfolder(trainedmodel, 'rsa'))):
                                                                    #if(True):
                                                                    if(default_run):
                                                                        rsa_models_comp(model, runinfo)
                                                                    else:
                                                                        print('rsa models comp already completed')

                                        if(do_analysis):

                                            if startheight == 'comp':
                                                runheight = True

                                            if runheight:
                                                if(False):
                                                    print('launching analysis of nodes\' generalizational capacity...')
                                                    generalization_main(model_to_analyse, runinfo)

                                                if(i==5 and control):
                                                #if(True):
                                                    if(False):
                                                        generalizations_comparisons_main(model, runinfo)

        else:
            print('analyzing models on specified regression task')

            base_model_path = os.path.join(runinfo.basefolder, 'ALL_%s' %model['typename'])

            all_regression_models = list(next(os.walk(base_model_path))[1])
            task_models = [x if '_r_' not in x for x in all_regression_models]
            regression_models = [x if '_r_' in x for x in all_regression_models]

            #all_regression_models_zipped = zip(task_models, regression_models)

            if do_data:

                print('saving hidden layer representations')

                for hpsname in all_regression_models:
                    
                    print('saving hidden layer representations for %s ...' %hpsname)

                    model_to_analyse = model.copy()
                    model_to_analyse['base'] = hpsname
                    model_to_analyse['model_path'] = os.path.join(base_model_path, hpsname)
                    model_to_analyse['path_to_config_file'] = os.path.join(base_model_path, hpsname, 'config.yaml')

                    modeloutputs_main(model_to_analyse, runinfo)

            if do_analysis:
                regression_task_cka_main(task_models, regression_models, runinfo, model.copy())




    print("Yay! Done, success")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Which parts of analysis to complete.')
    parser.add_argument('--data', type=bool, default = False, help='Extract data?')
    parser.add_argument('--results', type=bool, default=False, help='Fit TCs?')
    parser.add_argument('--analysis', type=bool, default=False, help='Analyze fitted TCs?')
    parset.add_argument('--regression_task', type=bool, default=False, help='Analyze models from regression task?')
    parser.add_argument('--S', type=bool, default=False, help='Include Spatial_temporal models?')
    parser.add_argument('--ST', type=bool, default=False, help='Include SpatioTemporal models?')
    parser.add_argument('--LSTM', type=bool, default=False, help='Include Spatial_temporal models?')

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

    main(args.data, args.results, args.analysis, args.regression_task, include)
# %%
