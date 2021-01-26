#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:27:05 2019

@author: Kai Sandbrink

"""

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import pickle, os, argparse
import multiprocessing as mp
import h5py
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression
import time
import resource
import os

# GLOBAL PARS
t_stride = 2
ntime=320
metrics = ['RMSE', 'r2', 'PCC']
nmetrics = len(metrics)
need_to_sleep = False
#datafolder = 'data/'

# %% UTILITY FUNCTIONS

def compute_metrics(y, pred):
    """ Computes RMSE, R2, and correlation performance metrics given true and predicted values 
    
    Parameters
    ----------
    y : np.array of floats, unit activations 
    pred : np.array of floats, corresponding predicted values
    
    Returns
    -------
    [rmse, r2, cor] : prediction metrics
    """
    try:
        rmse = np.sqrt(mean_squared_error(y.flatten(), pred.flatten()))
    except:
        print("MSE Error")
        print(pred)
        print(y)
    r2 = r2_score(y.flatten(), pred.flatten())
    cor = np.corrcoef(y.flatten(), pred.flatten())[0,1]
    
    #print(r2)
    assert r2 <=1 or np.isnan(r2), 'illegal r2 score!!!! %s %s %s' %(r2, y[:10], pred[:10])
    
    return [rmse, r2, cor]
    
def lstring(ilayer):
    if ilayer==-1:
        layer='data'
    else:
        layer = 'l%d' %ilayer
    return layer

def get_binidx(theta, nbins=8):
    binidx = int(( theta + np.pi ) / (2*np.pi/nbins))
    return binidx

def get_centers(fmapntime, ilayer, model, ntime = 320):
    """ Computes the temporal centers of the convolutional centers
    
    Parameters
    ----------
    fmapntime : int, length of temporal axis in given feature map
    ilayer : int, current layer of network, spindles = -1
    ntime : int, starting width
    model : dict -> Config, information of model
    
    Returns
    -------
    centers : list of ints, indices of temporal centers
    """
    
    #s_stride = model['s_stride']
    t_stride = model['t_stride']
    mtype = model['type']
    
    centers = np.arange(ntime)
    
    #if spindles skip
    #if(ilayer == -1):
    #    ilayer = 0
    
    for i in np.arange(0, ilayer + 1): #excludes spindles
        if(mtype == 'S' and i >= 4):
            centers = np.array([centers[i*t_stride] for i in range(int(np.ceil(len(centers)/t_stride)))])
        elif(mtype == 'ST'):
            centers = np.array([centers[i*t_stride] for i in range(int(np.ceil(len(centers)/t_stride)))])
    assert len(centers) == fmapntime, "Time dimensions mismatch!!!! centers: %d fmap: %d" %(len(centers), fmapntime)
    return centers

def read_layer_reps(ilayer, runinfo, model):
    '''Retrieve generated representations for the queried layer, adapted from Pranav 2020-12-01.
    
    Parameters
    ----------
    ilayer : int, index of current layer, spindles = -1
    runinfo : RunInfo, contains run configuration
    model : dict, contains model configuration
    
    Returns
    -------
    layer : array of floats [n_samples, n_features, n_time] or [n_samples, n_features, n_time, n_channels], 
        storing current layer activation
    '''
    
    layer = lstring(ilayer)
    
    if ilayer == -1:
        lo = pickle.load(open(os.path.join(runinfo.datafolder(model), layer + '.pkl'), 'rb'))
    else:
        try:
            lo = pickle.load(open(os.path.join(runinfo.datafolder(model), layer + '.pkl'), 'rb'))
        except:
            h5filepath = os.path.join(runinfo.datafolder(model), layer + '.hdf5')
        
            reps_layer = h5py.File(h5filepath, 'r')
            num_datasets = len(list(reps_layer))
            ds_shape = list(reps_layer.get('0').shape)
        
            batch_size = ds_shape[0]
            ds_shape[0] = batch_size*num_datasets
            lo = np.zeros((ds_shape))
            for i in range(num_datasets):
                lo[batch_size*i : batch_size*(i+1)] = reps_layer.get(str(i))[()]

    if(runinfo.verbose):
        print("read layer represenations. shape: ", lo.shape)

    return lo

# %% DIRECTION TUNING UTILITY FCTS

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
            
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    
    Arguments
    ---------
    v1 : np.array [2,]
    v2 : np.array [2,]
    
    Returns
    -------
    float, angle between v1 and v2
    """
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        #clip solves rounding errors

def angle_xaxis(v):
    """  Returns the angle between a vector and the x-axis ([1,0] vector) according to the rule
        Positive angle => positive y, negative angle => negative y
        Returns nan if v is 0
        
    Arguments
    ---------
    v : np.array [2,]
    
    Returns
    -------
    float, angle between v and the [1,0] vector
    """
    
    if not np.any(v):
        return np.nan
    elif v[1] >= 0:
        return angle_between(v, [1, 0])
    else:
        return - angle_between(v, [1,0])

def get_polar(X):
    ''' Converts Cartesian coordinates to polar
    
    Parameters
    ----------
    X : np.array [number of samples or timepoints, 2], Cartesian coordinates
    
    Returns
    -------
    Xp : np.array [number of samples or timepoints, 2], Polar coordinates
    '''
    
    Xp = np.zeros(X.shape)
    Xp[:,0] = np.apply_along_axis(np.linalg.norm, 1, X)
    Xp[:,1] = np.apply_along_axis(angle_xaxis, 1, X)
    return Xp

# %% DIRECTION AND VELOCITY TUNING FUNCTIONS
    
def linreg(X_train, X_test, Y_train, Y_test):
    ''' Performs linear regression with train and test set
    
    Parameters
    ----------
    X_train : np.array [number of samples,]
    X_test : np.array [number of samples,]
    Y_train : np.array [number of samples,]
    Y_test : np.array [number of samples,]
    
    Returns
    -------
    trainmets : list of floats [3], metrics for training fit specified in compute_metrics function
    testmets : list of floats [3], test metrics specified in compute_metrics function
    '''
    
    if(np.any(np.isnan(X_train))):
        print("Warning: Contains nans")
    c = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]
    trainmets = np.concatenate((compute_metrics(Y_train, X_train @ c), c, np.zeros(3 - len(c))))
    testmets = np.concatenate((compute_metrics(Y_test, X_test @ c), c, np.zeros(3 - len(c))))
    
    #print("LinReg Fit Score: " + str(trainmets[:3]))
    #print("LinReg Test Score: " + str(testmets[:3]))
    #print("Coefficients: " + str(c[:3]))
    return trainmets, testmets

def feature_set(Xcols_train, Xcols_test, Y_train, Y_test):
    ''' Combines different features together and excludes invalid values
    
    Arguments
    ---------
    Xcols_train : tuple of np.arrays of shape [nr of samples, k features]
    Xcols_test : tuple of np.arrays 
    Y_train : np.array
    Y_test : np.array
    
    Returns
    -------
    X_train : np.array [nr of samples, nr of features]
    X_test : np.array [nr of samples, nr of features]
    Y_train : np.array [nr of samples,]
    Y_test : np.array [nr of samples,]
    
    '''
    
    
    X_train = np.column_stack(Xcols_train).squeeze()
    X_test = np.column_stack(Xcols_test).squeeze()
    assert len(X_train) == len(Y_train), "Dimensions mismatch!!! X: %s , Y: %s" %(str(X_train.shape), str(Y_train.shape))
    if len(X_train.shape) > 1:
        nna = ~np.any(np.isnan(X_train), axis=1)
    else:
        nna = ~np.isnan(X_train)
    X_train = X_train[nna]
    Y_train = Y_train[nna]
    
    if len(X_test.shape) > 1:
        nna = ~np.any(np.isnan(X_test), axis=1)
    else:
        nna = ~np.isnan(X_test)
    X_test = X_test[nna]
    Y_test = Y_test[nna]
    
    return X_train, X_test, Y_train, Y_test

# %% DIRECTION AND VELOCITY TUNING 
def tune_row_vel(X, Y, row, isPolar = True):
    ''' Fit tuning curves for a single neuron / unit ("row") for velocity inputs (and similar kinematic features) for four different kinds of models
        1. simple linear regression over input features of X
        2. Directional tuning curves
        3. Velocity / Speed Dependence
        4. Direction x Velocity tuning curve
    
    Arguments
    ---------
    X : np.array [nr of samples, 2] 
    Y : np.array [nr of samples]
    row : int, row index
    isPolar : bool, is result already boolean?
    
    Returns
    -------
    tuple of 
        row : int, row index
        rowtraineval : np.array [4, 6] for four different model types. Cols 0-2: Metrics from compute_metrics, Cols 3-5: Linear Regression coeffs
        rowtesteval : np.array [4, 6] for four different model types. Cols 0-2: Metrics from compute_metrics, Cols 3-5: Linear regression coeffs
    '''

    if need_to_sleep:
        time.sleep(0.2)

        hard_limit = 1000*1024*1024
        soft, hard = resource.getrlimit(resource.RUSAGE_SELF)
        print("Soft and hard limits RUSAGE_SELF: %d, %d" %(soft, hard))
        resource.setrlimit(resource.RUSAGE_SELF, (hard_limit, hard_limit))
        soft, hard = resource.getrlimit(resource.RUSAGE_SELF)
        print("Soft and hard limits RUSAGE_SELF after mod: %d, %d" %(soft, hard))

        '''
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        print("Soft and hard limits  RLIMIT_CPU: %d, %d" %(soft, hard))
        resource.setrlimit(resource.RLIMIT_CPU, (soft // 2, hard  // 2))
        '''

        '''
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        print("Soft and hard limits  RLIMIT_DATA: %d, %d" %(soft, hard))
        resource.setrlimit(resource.RLIMIT_DATA, (hard_limit, hard_limit))
        '''
        '''

        soft, hard = resource.getrlimit(resource.RLIMIT_RSS)
        print("Soft and hard limits  RLIMIT_RSS: %d, %d" %(soft, hard))
        resource.setrlimit(resource.RLIMIT_RSS, (hard_limit, hard_limit))
        '''
        
        '''
        soft, hard = resource.getrlimit(resource.RLIMIT_NICE)
        print("Soft and hard limits  RLIMIT_NICE: %d, %d" %(soft, hard))
        resource.setrlimit(resource.RLIMIT_NICE, (15, 15))
        '''

        os.nice(19)
    
    rowtraineval = np.zeros((4,6))
    rowtesteval = np.zeros((4,6))
    #Axis 0: (0) training set (1) test set
    #test set els 4-6: linear regression coeffs
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    ##BASELINE LINREG MODEL
    #print("Baseline Linear Regression Model:")
    #Xtform_train = np.c_[X_train[:,0], X_train[:,1], np.ones_like(X_train[:,0])]
    #Xtform_test = np.c_[X_test[:,0], X_test[:,1], np.ones_like(X_test[:,0])]
    Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
            (X_train, np.ones_like(X_train[:,0])),
            (X_test, np.ones_like(X_test[:,0])),
            Y_train,
            Y_test
            )
    #print('Feature sets built')

    rowtraineval[0], rowtesteval[0] = linreg(Xtform_train, Xtform_test, Ytform_train, Ytform_test)

    #Change to polar coordinates
    if(not isPolar):
        #print("converting to polar...")
        X_train = get_polar(X_train)
        X_test = get_polar(X_test)

    if need_to_sleep:
        time.sleep(0.2)
    
    ##DIR DEP
    #print("Directional Dependence:")
    #Xtform_train = np.c_[np.cos(X_train[:,1]), np.sin(X_train[:,1]), np.ones_like(X_train[:,1])]
    #Xtform_test = np.c_[np.cos(X_test[:,1]), np.sin(X_test[:,1]), np.ones_like(X_test[:,1])]
    Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
        (np.cos(X_train[:,1]), np.sin(X_train[:,1]), np.ones_like(X_train[:,0])),
        (np.cos(X_test[:,1]), np.sin(X_test[:,1]), np.ones_like(X_test[:,0])),
        Y_train,
        Y_test
        )

    rowtraineval[1], rowtesteval[1] = linreg(Xtform_train, Xtform_test, Ytform_train, Ytform_test)

    if need_to_sleep:
        time.sleep(0.2)
    
    ##VEL DEP
    
    #Xtform_train = np.c_[X_train[:,0], np.ones_like(X_train[:,0])]
    #Xtform_test = np.c_[X_test[:,0], np.ones_like(X_test[:,0])]
    #print("Velocity Dependence:")
    Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
        (X_train[:,0], np.ones_like(X_train[:,0])),
        (X_test[:,0], np.ones_like(X_test[:,0])),
        Y_train,
        Y_test
        )
    rowtraineval[2], rowtesteval[2] = linreg(Xtform_train, Xtform_test, Ytform_train, Ytform_test)

    if need_to_sleep:
        time.sleep(0.2)
    
    ##DIR PLUS VEL DEP
    #Xtform_train = np.c_[X_train[:,0] * np.cos(X_train[:,1]), X_train[:,0] * np.sin(X_train[:,1]), np.ones_like(X_train[:,0])]
    #Xtform_test = np.c_[X_test[:,0] * np.cos(X_test[:,1]), X_test[:,0] * np.sin(X_test[:,1]), np.ones_like(X_test[:,0])]
    #print("Direction + Velocity Dependence:")   
    Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
        (X_train[:,0] * np.cos(X_train[:,1]), X_train[:,0] * np.sin(X_train[:,1]), np.ones_like(X_train[:,0])),
        (X_test[:,0] * np.cos(X_test[:,1]), X_test[:,0] * np.sin(X_test[:,1]), np.ones_like(X_test[:,0])),
        Y_train,
        Y_test
        )
    
    rowtraineval[3], rowtesteval[3] = linreg(Xtform_train, Xtform_test, Ytform_train, Ytform_test)

    if need_to_sleep:
        time.sleep(0.2)
        
    return (row, rowtraineval, rowtesteval)

# %% DECODING
def tune_row_decoding(X, Y, row):
    
    rowtraineval = np.zeros((1,3))
    rowtesteval = np.zeros((1,3))

    X_temp = X
    X = Y
    Y = X_temp

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    c = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]

    rowtraineval[0] = compute_metrics(Y_train, X_train @ c)
    rowtesteval[0] = compute_metrics(Y_test, X_test @ c)

    return (row, rowtraineval, rowtesteval)


# %% LABEL SPECIFICITY
    
def tune_row_label(X, Y, node):
    """Perform ANOVA-like analysis to determine how much of the variance can be
    explained by tuning to labels for a single unit ("row").
    
    Arguments
    ----------
    X: labels
    Y: node activitiy for each sample
    row: row index
    
    Returns
    ----------
    node: node index
    nodeeval: np.array of floats [21 x 3], return node evaluation.
        First 20 Rows: Metrics for each individual label [a, b, ...]:
            - Mean Squared (Deviation) of Group MSG
            - Mean Squared Within Group MSW
            - Explained Variance MSG/Var
        Row 20:
            - Total Mean
            - Total Variance
            - Total Fraction of Variance Explained
        Final Row: "Prediction" evaluation
            - RMSE
            - r2
            - PCC
    """
    
    X = label_binarize(X, np.unique(X))
    Y = Y.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    try:
        svm = OneVsRestClassifier(LinearSVC(max_iter=10, verbose=0))
        svm.fit(Y_train, X_train)
        
        nodetraineval = roc_auc_score(X_train, svm.decision_function(Y_train))
        try:
            nodetesteval = roc_auc_score(X_test, svm.decision_function(Y_test))
        except ValueError as err:
            print("test evaluation failed. %s" %err) 
            nodetesteval = 0.5
            
    except ValueError as err:
         print("fitting SVM failed. %s" %err) 
         nodetraineval = 0.5
        
    #print("Train eval: %s" %str(nodetraineval))
    #print("Test eval: %s" %str(nodetesteval))
    
    return node, np.array([(nodetraineval-0.5)*2]), np.array([(nodetesteval-0.5)*2])

# %% TUNE
        
def tune(X, fset, Y, centers, nmods, nmets, ilayer, mmod='std'):
    ''' Makes calls to tuning curves for individual rows using multiprocessing
       
       Arguments
       ---------
       X : np.array
       fset : string, name of kinematic feature
       Y : hidden layer activations to be fitted 
       centers : np.array of ints, mask for downsampling that contains the integer positions for temporal dimension that correspond to units in hidden layers
       nmods : number of different types of models that will be tested for this kinematic feature
       nmets : number of metrics to be used (usually 3 to correspond to output of compute_metrics)
       mmod : modifier for the kinematic feature type (default 'std')
       
       Returns
       -------
       trainevals : np.array [nr rows / units, nr of tuning curves, nr of metrics]
       testevals : np.array [nr rows / units, nr of tuning curves, nr of metrics]
       
        '''    
        
    Yshape = Y.shape
    if len(Yshape) == 3:
        ravelshape = (Y.shape[1], nmods, nmets)
        unravelshape = (Y.shape[1], nmods, nmets)
    else:
        ravelshape = (Y.shape[1] * Y.shape[3], nmods, nmets )
        unravelshape = (Y.shape[1], Y.shape[3], nmods, nmets)
    
    trainevals =  np.empty(ravelshape)
    testevals = np.empty(ravelshape)
        #Axis 0-k: lo dims except time
        #Axis k+1: (0) LinReg (1) Dir Tuning (2) Vel Tuning (3) Dir + Vel Tuning
        #Axis k+2: (0) RMSE (1) r2 (2) PCC
        
    #pool = mp.Pool(mp.cpu_count())
    #if(True):
    '''
    if (fset == 'ee' or fset == 'eepolar') and ilayer > 0:
        print("Proceeding without multiprocessing...")
        time.sleep(5)
        Y = Y.swapaxes(1,2).reshape((Y.shape[0], Y.shape[2], -1)).swapaxes(1,2)
        
        for irow in range(Y.shape[1]):
            print("Layer %d, Node %d / %d" %(ilayer, irow, Y.shape[1]))
            if(len(X.shape) > 1):
                x = X[..., centers]
            else:
                x = X
            
            y = Y[:,irow]
            
            ##RESHAPE FOR LINEAR REGRESSION
            
            if len(x.shape) > 1: ##FOR TIME-BASED DATA ( 2 COMPS PER TIMESPACE IS USE CASE)
                
                tcoff = sum(np.where(centers <= 32, True, False))
                x = x[...,tcoff:ntime-tcoff]
                y = y[:,tcoff:ntime-tcoff]
                x = x.swapaxes(1,2).reshape((-1,2))
            
            y = y.reshape((-1,))
            
            if fset == 'ee':
                isPolar = False
            else:
                isPolar = Tsrue
            
            _, trainevals[irow], testevals[irow] = tune_row_vel(x, y, irow, isPolar)    
    '''        
    #else:
    if(True):
        
        '''
        if fset == "labels" or fset == 'ee' or fset == 'eepolar':
            if ilayer == -1:
                n_cpus = 10
            elif ilayer == 0:
                n_cpus = 5
            else:
                n_cpus = 1
        else:
        '''

        if(True):
            n_cpus = 10
        print('Max CPU Count: %d , using %d ' %(mp.cpu_count(), n_cpus))
        pool = mp.Pool(n_cpus)
        
        results = []
        
        # Resize Y so that feature maps are appended as new rows in first feature map
        Y = Y.swapaxes(1,2).reshape((Y.shape[0], Y.shape[2], -1)).swapaxes(1,2)
    
        for irow in range(Y.shape[1]):
            #print("Layer %d, Node %d / %d" %(ilayer, irow, Y.shape[1]))

            if need_to_sleep:
                time.sleep(0.2)
            
            if(len(X.shape) > 1):
                x = X[..., centers]
            else:
                x = X
            
            y = Y[:,irow]
            
            ##RESHAPE FOR LINEAR REGRESSION
            
            if len(x.shape) > 1: ##FOR TIME-BASED DATA ( 2 COMPS PER TIMESPACE IS USE CASE)
                
                tcoff = sum(np.where(centers <= 32, True, False))
                x = x[...,tcoff:ntime-tcoff]
                y = y[:,tcoff:ntime-tcoff]
                x = x.swapaxes(1,2).reshape((-1,2))
            elif fset == 'labels':
                temp = np.ones_like(y)
                x = (temp.swapaxes(0,1) * x).swapaxes(0,1)
                x = x.reshape((-1,))
            y = y.reshape((-1,))

            if fset == 'acc':
                results.append(pool.apply_async(tune_row_vel, args=(x,y,irow, True)))
            elif fset == 'vel' or fset == 'eepolar' or fset == 'ang' or fset=='angvel':
                results.append(pool.apply_async(tune_row_vel, args=(x,y,irow, True)))
            elif fset == 'ee':
                results.append(pool.apply_async(tune_row_vel, args=(x,y,irow, False)))
                #sys.stdout.flush()
            elif fset == 'labels':
                results.append(pool.apply_async(tune_row_label, args=(x,y,irow)))
    
        results = [r.get() for r in results]
        
        pool.close()
        print("pool closed")
        pool.join()
        print("pool joined")
        
        for result in results:
            trainevals[result[0]] = result[1]
            testevals[result[0]] = result[2]

    #shape evals back to original shape
    trainevals = trainevals.reshape(unravelshape)
    testevals = testevals.reshape(unravelshape)
    
    return trainevals, testevals

def tune_decoding(X, fset, Y, centers, ilayer, mmod):

    if(len(X.shape) > 1):
        X = X[..., centers]
    
    # Resize Y so that feature maps are appended as new rows in first feature map
    Y = Y.swapaxes(1,2).reshape((Y.shape[0], Y.shape[2], -1)).swapaxes(1,2)

    # reshape so that both X and Y are in format [samples x timepoints, features]
    assert len(X.shape) > 1, "X has shape 1"
    
    X = X.swapaxes(1,2).reshape((-1, X.shape[1]))
    
    if len(Y.shape) > 2:
        Y = Y.swapaxes(1,2).reshape((-1, Y.shape[1]))
    else:
        Y = Y.reshape((-1, Y.shape[1]))

    #switch spindles to Y and neuron firing ratest to X
    X_temp = X
    X = Y
    Y = X_temp

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    if len(Y_train.shape) > 1:
        nna = ~np.any(np.isnan(Y_train), axis=1)
    else:
        nna = ~np.isnan(Y_train)
    X_train = X_train[nna]
    Y_train = Y_train[nna]
    
    if len(Y_test.shape) > 1:
        nna = ~np.any(np.isnan(Y_test), axis=1)
    else:
        nna = ~np.isnan(Y_test)
    X_test = X_test[nna]
    Y_test = Y_test[nna]

    trainevals = []
    testevals = []
    coefs = []
    
    assert Y_train.shape[1] > 1, 'Y is supposed to have multiple targets/columns'

    for feature in range(Y_train.shape[1]):

        lm = LinearRegression().fit(X_train, Y_train[feature])

        trainevals.append(compute_metrics(Y_train[feature], lm.predict(X_train)))
        testevals.append(compute_metrics(Y_test[feature], lm.predict(X_test)))

        coefs.append(lm.coef_.copy())

    trainevals = np.vstack(trainevals)
    testevals = np.vstack(testevals)
    coefs = np.vstack(coefs)

    return (trainevals, testevals, coefs)


def tune_layer(X, fset, xyplmvt, runinfo, ilayer, mmod, model, t_stride=2):
    """Performs several layer-wide calculations for fitting the tuning curves and saves output to file
    
    Arguments
    ---------
    X : np.array of floats, [nr_samples, nr_features, 320 timepoints]
    fset : string, kinematic tuning curve name
    xyplmvt : np.array of bools [nr_samples,], mask restricting to trials / movements that occur in the desired plane or orientation
    runinfo : RunInfo (dict extension)
    ilayer : int, index of current layer, -1 = spindles
    mmod : string, modifier to model or kinematic tuning curve type
    model : dict, information on model
    t_stride : int, temporal stride used when applying convolutional filters
    
    """

    savepath = runinfo.resultsfolder(model, fset)

        
    if mmod == 'decoding':
        savepath = runinfo.resultsfolder(model, '%s_%s' %(mmod, fset))
        
    try:
        os.makedirs(savepath)
    except:
        print('folder already exists')
    
    savename = 'l%d_%s_mets_%s_%s' %(ilayer+1, fset, mmod, runinfo.planestring())
    
    modelname = model['name']
    base = model['base']
    layer = lstring(ilayer)
    expid = runinfo['expid']
    
    #lo = pickle.load(open(os.path.join(runinfo.datafolder(model), layer + '.pkl'), 'rb'))
    lo = read_layer_reps(ilayer, runinfo, model)
    lo = lo[xyplmvt]
    
    centers = get_centers(lo.shape[2], ilayer, model)
    
    if mmod=='decoding':
        trainevals, testevals, coefs = tune_decoding(X, fset, lo, centers, ilayer, mmod)

        print('saving decoding reg coefs')
        np.save('%s/%s_coefs.npy' %(savepath, savename), coefs)        

    else:
        if (fset == 'vel' or fset == 'acc' or fset == 'eepolar' or fset == 'ang' or fset=='angvel' or fset=='ee'):
            nmods = 4
            nmets = 6
        elif (fset == 'labels'):
            nmods = 1
            nmets = 1
    
        if len(X) > 1:
            trainevals, testevals = tune(X, fset, lo, centers, nmods, nmets, ilayer, mmod)
        else:
            trainevals = np.empty((0, nmods, nmets))
            testevals = np.empty((0, nmods, nmets))
        
    print("Layer %d Completed!" %ilayer)
    
    np.save('%s/%s_train.npy' %(savepath, savename), trainevals)        
    np.save('%s/%s_test.npy' %(savepath, savename), testevals)
    print("files saved")

# %% DATA READIN
   
def X_data(fset = 'vel', runinfo = dict({'orientation': 'hor', 'plane': 'all', 'datafraction': 0.5}), 
           datafolder = '../data/spatial_temporal_4_8-16-16-32_64-64-64-64_5272/',
           threed= False,
           polar=True):
    """Returns input data for model. Uses global parameters.
    
    Returns
    ---------
    vels: np.array of floats [nsamples, 2, 320]
    xyplmvt: list of bools [nsamples_original], boolean mask to mask samples
        that are not in xy plane
    """
    
    ### INIT
    print(datafolder)
    try:
        kinvars = pd.read_hdf(os.path.join(datafolder, 'kinvars.hdf5')) 
        mc = np.swapaxes(kinvars['muscle_coords'].values, 0, 1)
        ee = np.swapaxes(kinvars['endeffector_coords'].values, 0, 1)
    except:
        print('using alternate method for accessing kinvars files by directly accessing needed arrays (pandas causes error)')
        filename = os.path.join(datafolder, 'kinvars.hdf5')
        
        
        with h5py.File(filename, "r") as f:
            
            key = 'data'
            data_group = 'block0_values'
            data_values = f[key][data_group][:]
            
            ee = data_values[:,:,:3]
            ee = np.swapaxes(ee, 1, 2)
            ee = np.swapaxes(ee, 0, 2)
            
            mc = data_values[:,:,7:32]
            mc = np.swapaxes(mc, 1, 2)
            mc = np.swapaxes(mc, 0, 2)
    
    vels = np.gradient(ee)[2]
        
    # EXCLUDE SAMPLES THAT AREN'T IN XY PLANE
    if(runinfo['orientation']=='hor'):
        if runinfo['height']=='all':
            xyplmvt = (1 - np.any(vels[:,2,:], axis=1)).astype('bool')
        else:
            xyplmvt = np.all(ee[:,2,:] == runinfo['height'], axis=1)          
        
        if not threed:
            ee = ee[:,:2]
            vels = vels[:,:2]
    else:
        if runinfo['height']=='all':
            xyplmvt = np.invert(np.any(vels[:,0,:], axis=1)) #movements in a yz plane
        else:
            xyplmvt = np.all(ee[:,0,:] == runinfo['height'], axis=1)          
        
        if not threed:
            ee = ee[:, 1:]
            vels = vels[:, 1:]
    
    ee = ee[xyplmvt]
    vels = vels[xyplmvt]
    X = vels

    if len(X) > 1:
        print(X.size)
        print(X.shape)

        if(fset == 'acc'):
            X = np.gradient(vels)[2]
            if polar:
                X = get_polar(X)
            #X = np.nan_to_num(X)
        elif(fset == 'ang'):
            Xa = np.apply_along_axis(angle_xaxis, 1, vels)
            Xb = np.gradient(Xa)[1]
            Xa = Xa[:,np.newaxis,:]
            Xb = Xb[:,np.newaxis,:]
            X = np.concatenate((Xb, Xa), axis=1)
        elif(fset == 'angvel'):
            X = np.apply_along_axis(angle_xaxis, 1, vels)
            X = np.gradient(X)[1]
            X = np.concatenate((X[:,np.newaxis,:], X[:,np.newaxis,:]), axis=1)
        elif(fset == 'labels'):
            X = pickle.load(open(os.path.join(datafolder, 'labels.pkl'), 'rb'))
            X = X[xyplmvt]
        elif(fset == 'ee'):
            X = ee
        elif(fset == 'eepolar'):
            x0 = ee[:,:,0]
            Xrel = [(ee[i].swapaxes(0,1) - x0[i]).swapaxes(0,1) for i in range(len(ee))]
            X[:,0,:] = np.apply_along_axis(np.linalg.norm, 1, Xrel)
            X[:,1,:] = np.apply_along_axis(angle_xaxis, 1, Xrel)
 
        elif(fset == 'vel'):
            
            if polar:
                X = get_polar(X)

    return X, xyplmvt

# %% MAIN

def main(fset, runinfo, model, startlayer=-1, endlayer=8, mmod='std'):

    assert fset == 'vel' or fset == 'acc' or fset == 'eepolar' or fset == 'ang'\
        or fset == 'labels' or fset=='angvel' or fset=='ee', "Invalid fset!!!"
        
    assert mmod == 'std' or mmod=='decoding', 'Invalid mmod!!!'
    
    modelname = model['name']
    nlayers = model['nlayers']
    base = model['base']
    
    print('evaluating %s model for %s %s, expid %d, plane %s ...' %(modelname, fset, mmod, runinfo['expid'], runinfo.planestring()))
    
    #print('runinfo datafolder', runinfo.datafolder(model))
    X, xyplmvt = X_data(fset, runinfo, datafolder=runinfo.datafolder(model))
    
    np.random.seed(42)
        
    for ilayer in np.arange(startlayer, min(endlayer, nlayers)):
        tune_layer(X, fset, xyplmvt, runinfo, ilayer, mmod, model)
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Tuning Properties Analysis for DeepDraw Models.')
    parser.add_argument('--fset', type=str, default = 'vel', help='Type of Features {"vel", "acc", "ang", "labels"}')
    parser.add_argument('--expid', type=int, help='Experiment ID Number')
    parser.add_argument('--startlayer', type=int, default=-1, help='Layer you want to start at')
    parser.add_argument('--endlayer', type=int, default=8, help='Layer you want to end before')
    parser.add_argument('--mmod', type=str, default='std', help='Model modifier')
    parser.add_argument('--mtype', type=str, default='S', \
                        help='Model type to be analyzed {(S)patial temporal, (T)emporal spatial, (S)patio(T)emporal, (F)ully (C)onnected}'
                        )
    
    args = parser.parse_args()
    
    main(args.fset, args.expid, args.startlayer, args.endlayer, args.mmod, args.mtype)