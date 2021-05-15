#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:50:22 2020

@author: kai
"""

import numpy as np
from kornblith_et_al_rsa_colab import *
from rowwise_neuron_curves_controls import lstring, read_layer_reps
import seaborn as sns
import os, pickle
import copy
import matplotlib.pyplot as plt

def cca(features_x, features_y):
    """Compute the mean squared CCA correlation (R^2_{CCA}).

    Args:
      features_x: A num_examples x num_features matrix of features.
      features_y: A num_examples x num_features matrix of features.

    Returns:
      The mean squared CCA correlations between X and Y.
    """
    qx, _ = np.linalg.qr(features_x)  # Or use SVD with full_matrices=False.
    qy, _ = np.linalg.qr(features_y)
    return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(features_x.shape[1], features_y.shape[1])

def layer_cka(ilayer, modelA, modelB, runinfo):
    #layer = lstring(ilayer)  
    #X = pickle.load(open(os.path.join(runinfo.datafolder(trainedmodel), layer + '.pkl'), 'rb'))
    X = read_layer_reps(ilayer, runinfo, modelA)
    X = X.reshape((X.shape[0], -1))                    
    #Y = pickle.load(open(os.path.join(runinfo.datafolder(controlmodel), layer + '.pkl'), 'rb'))
    Y = read_layer_reps(ilayer, runinfo, modelB)
    Y = Y.reshape((Y.shape[0], -1))
    
    print("Layer %d " %(ilayer + 1))
    print("X Shape: %s, Y Shape: %s" %(X.shape, Y.shape))
        
    cka_from_examples = cka(gram_linear(X), gram_linear(Y))

    return cka_from_examples

def models_cka_matrix(modelA, modelB, runinfo):
    '''Calculates a matrix of CKA scores comparing two models

    Arguments
    ---------
    modelA : dict containing model information for model A, including base and name
    modelB : dict containing model information for model B
    runinfo : RunInfo

    Returns
    -------
    cka_matrix : np.array of floats [1, nlayers + 1], cka scores for each layer
    '''

    nlayers = modelA['nlayers']
    cka_matrix = np.zeros((nlayers + 1,))

    for ilayer in np.arange(-1, nlayers):
        '''
        layer = lstring(ilayer)  
        #X = pickle.load(open(os.path.join(runinfo.datafolder(trainedmodel), layer + '.pkl'), 'rb'))
        X = read_layer_reps(ilayer, runinfo, modelA)
        X = X.reshape((X.shape[0], -1))                    
        #Y = pickle.load(open(os.path.join(runinfo.datafolder(controlmodel), layer + '.pkl'), 'rb'))
        Y = read_layer_reps(ilayer, runinfo, modelB)
        Y = Y.reshape((Y.shape[0], -1))
        
        print("Layer %d " %(ilayer + 1))
        print("X Shape: %s, Y Shape: %s" %(X.shape, Y.shape))
        
            
        cka_from_examples = cka(gram_linear(X), gram_linear(Y))
        '''

        cka_from_examples = layer_cka(ilayer, modelA, modelB, runinfo)
            
        cka_matrix[ilayer + 1] = cka_from_examples
    
    return cka_matrix


def main(trainedmodel, controlmodel, runinfo):
    '''Calculate the CKA and CCA scores of a given trained and control model for every layer and save output
    
    Arguments
    ---------
    trainedmodel : dict, information on model
    controlmodel : dict, information on model
    runinfo : RunInfo (extension of dict)
    
    '''
    
    nlayers = trainedmodel['nlayers']
    
    cka_matrix = np.zeros((1, nlayers + 1))
    cca_matrix = np.zeros((1, nlayers + 1))
    
    for ilayer in np.arange(-1, nlayers):
        layer = lstring(ilayer)  
        #X = pickle.load(open(os.path.join(runinfo.datafolder(trainedmodel), layer + '.pkl'), 'rb'))
        X = read_layer_reps(ilayer, runinfo, trainedmodel)
        X = X.reshape((X.shape[0], -1))                    
        #Y = pickle.load(open(os.path.join(runinfo.datafolder(controlmodel), layer + '.pkl'), 'rb'))
        Y = read_layer_reps(ilayer, runinfo, controlmodel)
        Y = Y.reshape((Y.shape[0], -1))
        
        print("Layer %d " %(ilayer + 1))
        print("X Shape: %s, Y Shape: %s" %(X.shape, Y.shape))
            
        cka_from_examples = cka(gram_linear(X), gram_linear(Y))
        cca_from_features = cca(X, Y)
            
        cka_matrix[0, ilayer + 1] = cka_from_examples
        cca_matrix[0, ilayer + 1] = cca_from_features
    
    folder = runinfo.analysisfolder(trainedmodel, 'rsa')
    os.makedirs(folder, exist_ok=True)
    
    np.save(os.path.join(folder, 'cka_matrix.npy'), cka_matrix)
    np.save(os.path.join(folder, 'cca_matrix.npy'), cca_matrix)
    
    sns.set()
    cka_ax = sns.heatmap(cka_matrix)
    fig = cka_ax.get_figure()
    fig.savefig(os.path.join(folder, 'cka.pdf'))
    
    fig.clf()
    cca_ax = sns.heatmap(cca_matrix)
    fig = cca_ax.get_figure()
    fig.savefig(os.path.join(folder, 'cca.pdf'))
    fig.clf()
    
    plt.close('all')
    
def rsa_models_comp(model, runinfo):
    ''' Combine the saved RSA scores of all implementations into a single plot
    
    Arguments
    ---------
    model : dict, information on model
    runinfo : RunInfo (extension of dict)
    '''
    
    nlayers = model['nlayers'] + 1
    
    modelbase = model['base']
    trainednamer = lambda i: modelbase + '_%d' %i
    modelnames = [trainednamer(i) for i in np.arange(1,6)]
        
    model = model.copy()
    
    cka = np.zeros((5, nlayers))
    cca = np.zeros((5, nlayers))
    
    for imodel, mname in enumerate(modelnames):
        model['name'] = mname
        model_cka = np.load(os.path.join(runinfo.analysisfolder(model, 'rsa'), 'cka_matrix.npy'))
        model_cca = np.load(os.path.join(runinfo.analysisfolder(model, 'rsa'), 'cca_matrix.npy'))
        
        cka[imodel] = model_cka
        cca[imodel] = model_cca
        
    folder = runinfo.sharedanalysisfolder(model, 'rsa')
    
    os.makedirs(folder, exist_ok=True)
    
    np.save(os.path.join(folder, 'cka_matrix.npy'), cka)
    np.save(os.path.join(folder, 'cca_matrix.npy'), cca)
    
    cka_ax = sns.heatmap(cka)
    cka_ax.set_xlabel('layer')
    cka_ax.set_ylabel('model')
    fig = cka_ax.get_figure()
    fig.savefig(os.path.join(folder, 'cka.pdf'))
    
    fig.clf()
    cca_ax = sns.heatmap(cca)
    cca_ax.set_xlabel('layer')
    cca_ax.set_ylabel('model')
    fig = cca_ax.get_figure()
    fig.savefig(os.path.join(folder, 'cca.pdf'))
    fig.clf()
    
    plt.close('all')
