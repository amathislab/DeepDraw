#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:06:05 2019

@author: kai

v6: wrapped in main, included abililty to process data/results/analysis separately
v5: change naming scheme to match with before/after controls
v4: dynamic checks if folder/files exist to see if should recalculate
v3: include individual planes
v2: restructured subfolders, expanded analysis

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
from combined_violinquantiles_controls import main as violin_main
from combined_violinquantiles_controls import comp_violin_main
from control_comparisons import main as comparisons_main
from control_comparisons import generalizations_comparisons_main
from control_comparisons import compare_all_types_main, invars_all_types_main
from prefdir_controls import main as prefdir_main
from generalization import main as generalization_main
from representational_similarity_analysis import main as rsa_main
from representational_similarity_analysis import rsa_models_comp
from polar_tcs import main as polar_tcs_main
from polar_tcs import complete_muscle_set

'''
mnames = ['spatial_temporal_4_8-16-16-32_64-64-64-64_5272',
    'temporal_spatial_4_16-16-32-64_64-64-64-64_5272',
    'spatiotemporal_4_8-8-32-64_7292'
    ]
'''

def format_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

# %% UTILS, CONFIG MODELS, AND GLOBAL VARS

#fsets = ['vel', 'acc', 'eepolar', 'labels', 'ang']
fsets = ['vel', 'acc', 'labels', 'ee']
decoding_fsets = ['ee', 'vel']
#expid = 100

orientations = ['hor', 'vert']
uniquezs = list(np.array([-45., -42., -39., -36., -33., -30., -27., -24., -21., -18., -15.,
                     -12.,  -9.,  -6.,  -3.,   0.,   3.,   6.,   9.,  12.,  15.,  18.,
                     21.,  24.,  27.,  30.]).astype(int))
uniquexs = list(np.array([ 6.,  9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42.,
                     45., 48., 51., 54.]).astype(int))
uniqueplanes=[uniquezs, uniquexs]

class RunInfo(dict):
    def __init__(self, *args, **kwargs):
        super(RunInfo, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    def planestring(self):
        return self.__dict__['orientation'] + str(self.__dict__['height'])
    
    def experimentfolder(self):
        return 'exp%d' %self.__dict__['expid']
    
    def resultsfolder(self, model, fset = None):
        resultsfolder =  'exp%d/results/%s/%s/%s/' %(self.__dict__['expid'], model['base'], model['name'], self.planestring())
        if fset is not None:
            resultsfolder = os.path.join(resultsfolder, fset)
        return resultsfolder
    
    def sharedanalysisfolder(self, model, analysis = None, sp=True):
        if sp:
            sharedanalysisfolder = 'exp%d/analysis/%s/comparison/%s/' %(self.__dict__['expid'], model['base'], self.planestring())
        else:
            sharedanalysisfolder = 'exp%d/analysis/%s/comparison/' %(self.__dict__['expid'], model['base'])
        if analysis is not None:
            sharedanalysisfolder = os.path.join(sharedanalysisfolder, analysis)
        return sharedanalysisfolder
    
    def baseanalysisfolder(self, model, analysis=None):
        baseanalysisfolder = 'exp%d/analysis/%s/%s/' %(self.__dict__['expid'], model['base'], model['name'])
        if analysis is not None:
            baseanalysisfolder = os.path.join(baseanalysisfolder, analysis)
        return baseanalysisfolder
    
    def analysisfolder(self, model, analysis=None):
        baseanalysisfolder = self.baseanalysisfolder(model)
        analysisfolder = os.path.join(baseanalysisfolder, self.planestring())
        if analysis is not None:
            analysisfolder = os.path.join(analysisfolder, analysis)
        return analysisfolder
    
    def generalizationfolder(self, model, analysis=None):
        generalizationfolder = os.path.join(self.baseanalysisfolder(model), 'generalization')
        if analysis is not None:
            generalizationfolder = os.path.join(generalizationfolder, analysis)
        return generalizationfolder
    
    def datafolder(self, model = None):
        datafolder = 'exp%d/data/' %self.__dict__['expid']
        if model is not None:
            datafolder = os.path.join(datafolder , model['base'], model['name'])
        return datafolder
    
    def allmodelfolder(self, analysis = None, sp=True):
        allmodelfolder = 'exp%d/analysis/combined/' %self.__dict__['expid']
        if sp:
            allmodelfolder = os.path.join(allmodelfolder, self.planestring())
        if analysis is not None:
            allmodelfolder = os.path.join(allmodelfolder, analysis)
        return allmodelfolder
    
# %% EXPERIMENTAL RUN CONFIG

runinfo = RunInfo({'expid': 102, #internal experiment id
                   'datafraction': 0.5,
                   'randomseed': 2000,
                   'randomseed_traintest': 42,
                   'dirr2threshold': 0.2,
                   'verbose': 0,
                   'model_experiment_id': 4 #as per Pranav's model generation
            })
    
"""
experimental log:
expid: 100
    - datafraction 1/10
expid: 101
    - datafraction 0.2
"""
    
# %% SAVE OUTPUTS AND RUN ANALYSIS

def main(do_data=False, do_results=False, do_analysis=False, include = ['S', 'T','ST']):
    
    matplotlib.pyplot.rcParams.update({'legend.title_fontsize':40})
    
    print('experimental id %d' %runinfo['expid'])
    print('data fraction %.2f' %runinfo['datafraction'])
    
    os.makedirs(runinfo.experimentfolder(), exist_ok=True)
    configfilename = os.path.join(runinfo.experimentfolder(), 'config.yaml')
    with io.open(configfilename, 'w',  encoding='utf8') as outfile:
        yaml.dump(runinfo, outfile, default_flow_style=False, allow_unicode=True)
        
    allmodels = [dict({'type': 'S',
            'base': 'spatial_temporal_4_8-16-16-32_64-64-64-64_5272',
            #'name': 'spatial_temporal_4_8-16-16-32_64-64-64-64_5272',
            'nlayers': 8,
            'max_act': 14,
            'control': False,
            'cmap': 'Blues_r',
            'color': 'C0',
            'control_cmap': 'Purples_r'}),
      dict({'type': 'T',
            'base': 'temporal_spatial_4_16-16-32-64_64-64-64-64_5272',
            #'name': 'temporal_spatial_4_16-16-32-64_64-64-64-64_5272',
            'nlayers': 8,
            'max_act': 14,
            'control': False,
            'cmap': 'Oranges_r',
            'color': 'orange',
            'control_cmap': 'Reds_r'}),
        dict({'type': 'ST',
              'base': 'spatiotemporal_4_8-8-32-64_7272',
            #  'name': 'spatiotemporal_4_8-8-32-64_7292',
            'nlayers': 4,
            'max_act': 14,
            'control': False,
            'cmap': 'Greens_r',
            'color': 'green',
            'control_cmap': 'Greys_r'})]

    models = [model for model in allmodels if (model['type'] in include)]
    #print(models)

    
    for imodel, model in enumerate(models):
        
        for i in np.arange(1, 6):
            
            #runinfo['orientation'] = 'hor'
            #runinfo['height'] = 'all'
    
            modelname = model['base'] + '_%d' %i            
            model['name'] = modelname
            
            model_to_analyse = model.copy()
            trainedmodel = model.copy()
    
            for control in [False, True]:
                
                if control:
                    modelname = modelname + 'r'
                    model_to_analyse['name'] = modelname
                    model_to_analyse['control'] = True
                
                if(do_data):
                    if(not os.path.exists(runinfo.datafolder(model_to_analyse))):
                        print('generating output for model %s ...' %modelname)
                        modeloutputs_main(model_to_analyse, runinfo)    
                    else:
                        print('data for model %s already generated' %modelname)
                
                if(do_results or do_analysis):
                    for ior, orientation in enumerate(orientations):
                        runinfo['orientation'] = orientation
                        
                        for height in (['all'] + uniqueplanes[ior]):
                            runinfo['height'] = height
                            #print(height)
                            
                            runinfo_to_analyse = copy.copy((runinfo))
                         
                            if(do_results):
                                print('running analysis for model %s plane %s...' %(modelname, runinfo.planestring()))
                
                                for fset in fsets:
                                    
                                    if(not os.path.exists(runinfo.resultsfolder(model_to_analyse, fset))):
                                        #or fset == 'labels'):
                                        
                                        print('running %s analysis for model %s plane %s...' %(fset, modelname, runinfo.planestring()))
                                        tuningcurves_main(fset,
                                                          runinfo_to_analyse,
                                                          model_to_analyse,
                                                          )
                                    
                                    else:
                                        print('%s analysis for model %s plane %s already completed' %(fset, modelname, runinfo.planestring()))
                            
                            """
                            print('running decoding analysis for model %s plane %s...' %(modelname, runinfo.planestring()))
                            
                            for fset in decoding_fsets:
                                #if(not os.path.exists(runinfo.resultsfolder(model_to_analyse, 'decoding_%s' %fset))):
            
                                if(True):
                                   print('running %s decoding analysis for model %s plane %s...' %(fset, modelname, runinfo.planestring()))
                                   tuningcurves_main(fset,
                                                      runinfo_to_analyse,
                                                      model_to_analyse,
                                                      mmod='decoding'
                                                      )
                                else:
                                    print('decoding analysis already completed')
                            """
                            
                            if(do_analysis):
                                print('compiling results and generating graphs for model %s plane %s...' %(modelname, runinfo.planestring()))
                                if(not os.path.exists(runinfo.analysisfolder(model_to_analyse, 'joint'))):
                                #if(True):
                                    violin_main(model_to_analyse, runinfo_to_analyse)
                                else:
                                    print('violin plot already exists')
                                
                                """
                                print('generating scores vs var plot for model %s plane %s...' %(modelname, runinfo.planestring()))
                                if(not os.path.exists(runinfo.analysisfolder(model_to_analyse, 'scorevsvar'))):
                                    scorevsvar(model_to_analyse, runinfo_to_analyse)
                                else:
                                    print('scores vs var plot already exists')
                                """
                                
                                print('generating polar tuning curve plots for model %s plane %s ...' %(modelname, runinfo.planestring()))
                                if(not os.path.exists(runinfo.analysisfolder(model_to_analyse, 'polar_tcs'))):
                                #if(True):
                                    polar_tcs_main(model_to_analyse, runinfo_to_analyse)
                                else:
                                    print('polar tc plots already exist')
                                
                                '''
                                print('making complete muscle set of tuning curves...')
                                if(not control and i == 1 and model_to_analyse['type'] == 'S'):
                                    complete_muscle_set(model_to_analyse, runinfo_to_analyse)
                                '''
                                
                                #print('generating preferred direction histograms for model %s plane %s...' %(modelname, runinfo.planestring()))
                                if(not os.path.exists(runinfo.analysisfolder(model_to_analyse, 'prefdir'))):
                                #if(True):
                                    prefdir_main(model_to_analyse, runinfo_to_analyse)
                                else:
                                    print('pref dir plots already exist')
                                
                                '''
                                print('generating tuning curve plots for optimal neurons for model %s plane %s...' %(modelname, runinfo.planestring()))
                                if(not os.path.exists(runinfo.analysisfolder(model_to_analyse, 'polar_tcs'))):
                                    polar_tcs_main(model_to_analyse, runinfo_to_analyse)
                                else:
                                    print('polar tc plots already exist')
                                '''
                                
                                """
                                print('decoding trajectories for model %s plane %s...' %(modelname, runinfo.planestring()))
                                #if(not np.all(np.array([os.path.exists(runinfo.analysisfolder(model_to_analyse, '%s_decoding')) for fset in decfsets])):
                                if(True):
                                    decoding_main(model_to_analyse, runinfo_to_analyse)
                                else:
                                    print('decoding analysis already completed')
                                """
                                
                                """
                                if control:
                                     #if(not np.all(np.array([os.path.exists(runinfo.analysisfolder(model, '1v1comp')) for fset in decfsets])):
                                     1v1comp
                                """
                                
                                if(control):
                                    if(not os.path.exists(runinfo.analysisfolder(trainedmodel, 'comp_violin'))):
                                    #if(True):
                                        print('saving comparison violin plot for model %s plane %s...' %(modelname, runinfo.planestring()))
                                        comp_violin_main(trainedmodel, model_to_analyse, runinfo)
                                    else:
                                        print('comparison violin plot already saved')
                                    # if(not os.path.exists(runinfo.sharedanalysisfolder(trainedmodel, 'kindiffs_plots'))):
                                    
                                    if(runinfo.planestring() == 'horall'):
                                        if(not os.path.exists(runinfo.analysisfolder(trainedmodel, 'rsa'))):
                                        #if(True):
                                            print('computing rsa for model %s plane %s ... ' %(modelname, runinfo.planestring()))
                                            rsa_main(trainedmodel, model_to_analyse, runinfo)
                                       
                                        else:
                                            print('rsa already saved')
                                
                                if (i==5 and control):
                                    comparisons_main(model, runinfo)
                                    
                                    #if(imodel == len(models) - 1):
                                    if(imodel == 3):
                                        compare_all_types_main(allmodels, runinfo)
                                        
                                    if(runinfo.planestring() == 'horall'):
                                        print('combining rsa results for all models')
                                        if(not os.path.exists(runinfo.sharedanalysisfolder(trainedmodel, 'rsa'))):
                                            rsa_models_comp(model, runinfo)
                                        else:
                                            print('rsa models comp already completed')
    
                if(do_analysis):
                    #if(not os.path.exists(runinfo.generalizationfolder(model))):
                    print('launching analysis of nodes\' generalizational capacity...')
                    generalization_main(model_to_analyse, runinfo)
                    
                    if(i==5 and control):
                    #if(True):
                        generalizations_comparisons_main(model, runinfo)
                        
    if(do_analysis):
        invars_all_types_main(allmodels, runinfo)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Which parts of analysis to complete.')
    parser.add_argument('--data', type=bool, default = False, help='Extract data?')
    parser.add_argument('--results', type=bool, default=False, help='Fit TCs?')
    parser.add_argument('--analysis', type=bool, default=False, help='Analyze fitted TCs?')
    parser.add_argument('--S', type=bool, default=False, help='Include Spatial_temporal models?')
    parser.add_argument('--T', type=bool, default=False, help='Include Temporal_spatial models?')
    parser.add_argument('--ST', type=bool, default=False, help='Include SpatioTemporal models?')
    
    args = parser.parse_args()
    
    include = []
    if args.S:
        include.append('S')
    if args.T:
        include.append('T')
    if args.ST:
        include.append('ST')
    if (include == []):
        include = ['S', 'T', 'ST']
    
    main(args.data, args.results, args.analysis, include)
