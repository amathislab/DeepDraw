#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:27:05 2019

@author: Kai Sandbrink

===CONTROLS LOG===
v6: debug position r2 scores
v5: include position
v4: changed label specificity measure (from LR to MV ROC AUC)
v3: individual planes
v2: restructured subfolders

===ROWWISE LOG===
v6: Make model name variable
v5_vonmises: Add von Mises function
v5: add angular velocity and label specificity, velocity in polar purely,
    allowed for analysis in vertical planes
v4: switch to performing linear regression manually
v3: eepolar fset, exclude "undefined theta" points from evaluation score
    removed standardization of inputs (important!)
v2: limit to central 80% of time row, now unravels eval to original shape,
    save training sets as well, include evaluation fset

"""

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.utils import resample
from scipy.optimize import minimize
from scipy.stats import f_oneway
import numpy as np
import pandas as pd
import pickle, os, argparse, sys
import multiprocessing as mp
import h5py
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier

# GLOBAL PARS
#t_kernelsize = 7
t_stride = 2
ntime=320
metrics = ['RMSE', 'r2', 'PCC']
nmetrics = len(metrics)

#datafolder = '../data/'
datafolder = 'data/' #CHANGED DATA FOLDER FOR CONTROLS

#mmod = 'rayleigh'

# %% UTILITY FUNCTIONS

def compute_metrics(y, pred):
    """ Computes various different performance metrics given true and predicted values """
    try:
        rmse = np.sqrt(mean_squared_error(y.flatten(), pred.flatten()))
    except:
        print("MSE Error")
        print(pred)
        print(y)
    r2 = r2_score(y.flatten(), pred.flatten())
    cor = np.corrcoef(y.flatten(), pred.flatten())[0,1]
    
    print(r2)
    assert r2 <=1 , 'illegal r2 score!!!! %s %s' %(y[:10], pred[:10])
    
    return [rmse, r2, cor]
    
def lstring(ilayer):
    if ilayer==-1:
        layer='data'
    else:
        layer = 'l%d' %ilayer
    return layer

def get_binidx(theta, nbins=8):
    
    #binlims = [-3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    binidx = int(( theta + np.pi ) / (2*np.pi/nbins))
    return binidx

def get_centers(fmapntime, ntime = 320, t_stride = 2):
    centers = np.arange(ntime)
    #fmapntime = lo.shape[2]
    #if fmapntime != len(centers):
    for i in range(int(np.log2(len(centers)/ fmapntime))):
        centers = np.array([centers[i*t_stride] for i in range(len(centers)//2)])
    assert len(centers) == fmapntime, "Time dimensions mismatch!!!!"
    return centers

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
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        #clip solves rounding errors

def angle_xaxis(v):
    """ Positive angle => positive y, negative angle => negative y
        Returns nan if v is 0
    """
    if not np.any(v):
        return np.nan
    elif v[1] >= 0:
        return angle_between(v, [1, 0])
    else:
        return - angle_between(v, [1,0])

def get_polar(X):
    Xp = np.zeros(X.shape)
    #print("Xp shape: ", Xp.shape)
    #print("linalg: ",  np.apply_along_axis(np.linalg.norm, X, 1).shape)
    #print(X[0])
    Xp[:,0] = np.apply_along_axis(np.linalg.norm, 1, X)
    Xp[:,1] = np.apply_along_axis(angle_xaxis, 1, X)
    return Xp

# %% DIRECTION AND VELOCITY TUNING FUNCTIONS
    
def linreg(X_train, X_test, Y_train, Y_test):
    if(np.any(np.isnan(X_train))):
        print("Warning: Contains nans")
    c = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]
    trainmets = np.concatenate((compute_metrics(Y_train, X_train @ c), c, np.zeros(3 - len(c))))
    testmets = np.concatenate((compute_metrics(Y_test, X_test @ c), c, np.zeros(3 - len(c))))
    
    print("LinReg Fit Score: " + str(trainmets[:3]))
    print("LinReg Test Score: " + str(testmets[:3]))
    print("Coefficients: " + str(c[:3]))
    return trainmets, testmets

def feature_set(Xcols_train, Xcols_test, Y_train, Y_test):
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
    
    """
    if len(Y_test.shape) > 1:
        nna = ~np.any(np.isnan(Y_test), axis=1)
    else:
        nna = ~np.isnan(Y_test)
    X_test = X_test[nna]
    Y_test = Y_test[nna]
    
    if len(Y_train.shape) > 1:
        nna = ~np.any(np.isnan(Y_train), axis=1)
    else:
        nna = ~np.isnan(Y_train)
    X_train = X_train[nna]
    Y_train = Y_train[nna]
    """
    return X_train, X_test, Y_train, Y_test

# %% DIRECTION AND VELOCITY TUNING 
def tune_row_vel(X, Y, row, isPolar = True):
    
    rowtraineval = np.zeros((4,6))
    rowtesteval = np.zeros((4,6))
    #Axis 0: (0) training set (1) test set
    #test set els 4-6: linear regression coeffs
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    ##BASELINE LINREG MODEL
    print("Baseline Linear Regression Model:")
    #Xtform_train = np.c_[X_train[:,0], X_train[:,1], np.ones_like(X_train[:,0])]
    #Xtform_test = np.c_[X_test[:,0], X_test[:,1], np.ones_like(X_test[:,0])]
    Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
            (X_train, np.ones_like(X_train[:,0])),
            (X_test, np.ones_like(X_test[:,0])),
            Y_train,
            Y_test
            )

    rowtraineval[0], rowtesteval[0] = linreg(Xtform_train, Xtform_test, Ytform_train, Ytform_test)
    
    #Change to polar coordinates
    if(not isPolar):
        print("converting to polar...")
        X_train = get_polar(X_train)
        X_test = get_polar(X_test)
    
    ##DIR DEP
    print("Directional Dependence:")
    #Xtform_train = np.c_[np.cos(X_train[:,1]), np.sin(X_train[:,1]), np.ones_like(X_train[:,1])]
    #Xtform_test = np.c_[np.cos(X_test[:,1]), np.sin(X_test[:,1]), np.ones_like(X_test[:,1])]
    Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
        (np.cos(X_train[:,1]), np.sin(X_train[:,1]), np.ones_like(X_train[:,0])),
        (np.cos(X_test[:,1]), np.sin(X_test[:,1]), np.ones_like(X_test[:,0])),
        Y_train,
        Y_test
        )

    rowtraineval[1], rowtesteval[1] = linreg(Xtform_train, Xtform_test, Ytform_train, Ytform_test)
    
    ##VEL DEP
    
    #Xtform_train = np.c_[X_train[:,0], np.ones_like(X_train[:,0])]
    #Xtform_test = np.c_[X_test[:,0], np.ones_like(X_test[:,0])]
    print("Velocity Dependence:")
    Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
        (X_train[:,0], np.ones_like(X_train[:,0])),
        (X_test[:,0], np.ones_like(X_test[:,0])),
        Y_train,
        Y_test
        )
    rowtraineval[2], rowtesteval[2] = linreg(Xtform_train, Xtform_test, Ytform_train, Ytform_test)
    
    ##DIR PLUS VEL DEP
    #Xtform_train = np.c_[X_train[:,0] * np.cos(X_train[:,1]), X_train[:,0] * np.sin(X_train[:,1]), np.ones_like(X_train[:,0])]
    #Xtform_test = np.c_[X_test[:,0] * np.cos(X_test[:,1]), X_test[:,0] * np.sin(X_test[:,1]), np.ones_like(X_test[:,0])]
    print("Direction + Velocity Dependence:")   
    Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
        (X_train[:,0] * np.cos(X_train[:,1]), X_train[:,0] * np.sin(X_train[:,1]), np.ones_like(X_train[:,0])),
        (X_test[:,0] * np.cos(X_test[:,1]), X_test[:,0] * np.sin(X_test[:,1]), np.ones_like(X_test[:,0])),
        Y_train,
        Y_test
        )
    
    rowtraineval[3], rowtesteval[3] = linreg(Xtform_train, Xtform_test, Ytform_train, Ytform_test)
        
    return (row, rowtraineval, rowtesteval)

# %% KIN DECODING

def tune_kin_decoding(X, Y, kin):
    
    rowtraineval = np.zeros((1,3))
    rowtesteval = np.zeros((1,3))
    #Axis 0: (0) training set (1) test set
    #test set els 4-6: linear regression coeffs
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    ##BASELINE LINREG MODEL
    print("Baseline Linear Regression Model:")
    #Xtform_train = np.c_[X_train[:,0], X_train[:,1], np.ones_like(X_train[:,0])]
    #Xtform_test = np.c_[X_test[:,0], X_test[:,1], np.ones_like(X_test[:,0])]
    """Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
            X_train,
            X_test,
            Y_train,
            Y_test
            )
    """
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
    
    """
    if len(Y_test.shape) > 1:
        nna = ~np.any(np.isnan(Y_test), axis=1)
    else:
        nna = ~np.isnan(Y_test)
    X_test = X_test[nna]
    Y_test = Y_test[nna]
    
    if len(Y_train.shape) > 1:
        nna = ~np.any(np.isnan(Y_train), axis=1)
    else:
        nna = ~np.isnan(Y_train)
    X_train = X_train[nna]
    Y_train = Y_train[nna]
    """
    Y_train = np.where(~np.isnan(Y_train), Y_train, 0)
    Y_test = np.where(~np.isnan(Y_test), Y_test, 0)
    
    """
    c = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]
    trainmets = compute_metrics(Y_train, X_train @ c)
    testmets = compute_metrics(Y_test, X_test @ c)
    """
    
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    rm = Ridge(alpha=1)
    rm.fit(X_train, Y_train)
    
    trainmets = compute_metrics(Y_train, rm.predict(X_train))
    testmets = compute_metrics(Y_test, rm.predict(X_test))

    rowtraineval[0], rowtesteval[0] = trainmets, testmets
    
    print(rowtraineval)
    print(rowtesteval)
    
    return (kin, rowtraineval, rowtesteval, rm)

# %% LABEL SPECIFICITY
    
def tune_row_label(X, Y, node):
    """Perform ANOVA-like analysis to determine how much of the variance can be
    explained by tuning to labels.
    
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
    
    #print(X, Y)
    #print(X.shape)
    #X = label_binarize(X, list(range(20)))
    

    X = label_binarize(X, np.unique(X))
    Y = Y.reshape(-1,1)
    #print(X.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    #svm = OneVsRestClassifier(SVC(kernel='linear'))
    try:
        svm = OneVsRestClassifier(LinearSVC(max_iter=10, verbose=0))
        #svm = LinearSVC(max_iter = 10, verbose=1)
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
        
    print("Train eval: %s" %str(nodetraineval))
    print("Test eval: %s" %str(nodetesteval))
    
    #return node, np.array([np.abs(nodetraineval-0.5)*2]), np.array([np.abs(nodetesteval-0.5)*2])
    return node, np.array([(nodetraineval-0.5)*2]), np.array([(nodetesteval-0.5)*2])
    
    #nodetraineval = np.zeros((22, 3))
    #nodetesteval = np.zeros((22, 3))
    
    """
    nodetraineval = np.zeros((22, 3))
    nodetesteval = np.zeros((22, 3))

    ##TRAINING SET
    groups = [Y_train[X_train == ichar] for ichar in range(20)]
    groupmeans = [group.mean() for group in groups]
    
    var = np.var(Y_train)
    mean = np.mean(Y_train)
    n = len(Y_train)
    
    MSGs = [(groupmean - mean) ** 2 for groupmean in groupmeans]
    MSWs = [np.var(group) for group in groups]
        #to recover fstat from MSGs, MSWs:
        # fstat = (MSGs / 19) / (MSWs / (len(Y) - 20))
    frex = [(len(groups[i]) / n ) * MSGs[i]/var for i in range(len(groups))]
    
    #print(len(MSGs), len(nodeeval[:20,0]))
    
    nodetraineval[:20, 0] = MSGs
    nodetraineval[:20, 1] = MSWs
    nodetraineval[:20, 2] = frex
        
    nodetraineval[20, 0] = Y_train.mean()
    nodetraineval[20, 1] = var
    nodetraineval[20, 2] = sum(frex)
    
    ypred = np.array([groupmeans[int(x)] for x in X_train])
    nodetraineval[21] = compute_metrics(Y_train, ypred)
    
    
    ## TEST SET

    groups = [Y_test[X_test == ichar] for ichar in range(20)]
    groupmeans = [group.mean() for group in groups]
    
        
    var = np.var(Y_test)
    mean = np.mean(Y_test)
    n = len(Y_test)
    
    MSGs = [(groupmean - mean) ** 2 for groupmean in groupmeans]
    MSWs = [np.var(group) for group in groups]
        #to recover fstat from MSGs, MSWs:
        # fstat = (MSGs / 19) / (MSWs / (len(Y) - 20))
    frex = [(len(groups[i]) / n ) * MSGs[i]/var for i in range(len(groups))]
    
    #print(len(MSGs), len(nodeeval[:20,0]))
    
    nodetesteval[:20, 0] = MSGs
    nodetesteval[:20, 1] = MSWs
    nodetesteval[:20, 2] = frex
        
    nodetesteval[20, 0] = Y_test.mean()
    nodetesteval[20, 1] = var
    nodetesteval[20, 2] = sum(frex)
  
    ypred = np.array([groupmeans[int(x)] for x in X_test])
    nodetesteval[21] = compute_metrics(Y_test, ypred)

    print('Mean, variance, total explained for row: ' + str(nodetesteval[20]))
    print("Label-prediction evaluation metrics for row: " + str(nodetesteval[21])) 
    
    return (node, nodetraineval, nodetesteval)
    """

# %% von mises functions

def vonmises(x, mu, kappa, alpha, beta):
    vonmises = alpha * np.exp(kappa * x - mu) + beta
    return vonmises

def jac_vonmises(x, mu, kappa, alpha, beta):
    """Jacobian with respect to model parameters (mu, kappa, alpha, beta)
        averaged over data set
        """
    #print("x: " + str(x))
   # print(mu)
    dmu = alpha * kappa * np.sin(x - mu) * np.exp(kappa * np.cos(x - mu))
    dkappa = alpha * np.cos(x - mu) * np.exp(kappa * np.cos(x - mu))
    dalpha = np.exp(kappa * np.cos(x - mu))
    dbeta = np.ones_like(dmu)
    
    #jac = [dmu, dkappa, dalpha, dbeta]    
    #print(jac)
    return np.c_[dmu, dkappa, dalpha, dbeta]

def vonmises_loss(X, Y, mu, kappa, alpha, beta):
    #print("X shape: " + str(X.shape))
    #print("Ytrue shape: " + str(Y.shape))
    ypred = vonmises(X, mu, kappa, alpha, beta)
    #print("Ytrue shape: " + str(Y.shape))
    #print("Ypred shape: " + str(ypred.shape))
    return compute_metrics(Y, ypred)

def jac_vonmises_loss(X, Y, mu, kappa, alpha, beta):
    rmse = vonmises_loss(X, Y, mu, kappa, alpha, beta)[0]
    ypred = vonmises(X, mu, kappa, alpha, beta)
    diff = Y - ypred
    jac = jac_vonmises(X, mu, kappa, alpha, beta)
    jac_loss = (1 / rmse) * np.mean(((- jac).swapaxes(0,1) * diff).swapaxes(0,1), axis=0)
    return jac_loss

def vonmises_wrapper(pars, X, Y):
    #print(pars)
    rmse = vonmises_loss(X, Y, pars[0], pars[1], pars[2], pars[3])[0]
    return rmse

def jac_vonmises_wrapper(pars, X, Y):
    jac = jac_vonmises_loss(X, Y, pars[0], pars[1], pars[2], pars[3])
    return jac

def vonmises_dirtune(X_train, X_test, Y_train, Y_test):
    x0 = [X_train.mean(), X_train.std(), Y_train.std(), Y_train.mean()]
    optimization = minimize(vonmises_wrapper, x0, args=(X_train, Y_train), 
                #method='BFGS', jac=jac_vonmises_wrapper)
                method='BFGS', jac=False)
    #optimization = minimize(vonmises_wrapper, x0, args=(X_train, Y_train),
    #            method = 'Nelder-Mead')
    print(optimization)
    mu, kappa, alpha, beta = optimization['x']
    
    coeffs = [mu, kappa, alpha, beta]
    print("Coefficients: " + str(coeffs))
    
    ypred = vonmises(X_train, mu, kappa, alpha, beta)
    traineval = compute_metrics(Y_train, ypred)
    print("Training Loss: " + str(traineval))
    ypred = vonmises(X_test, mu, kappa, alpha, beta)
    testeval = compute_metrics(Y_test, ypred)
    print("Test Loss: " + str(testeval))
    
    return traineval + coeffs, testeval + coeffs
    

# %% TUNE ROW VON MISES
def tune_row_vonmises(X, Y, node, isPolar = True):

    rowtraineval = np.zeros((2,7))
    rowtesteval = np.zeros((2,7))
    #Axis 0: (0) training set (1) test set
    #test set els 4-6: linear regression coeffs
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    ##BASELINE LINREG MODEL
    print("Baseline Linear Regression Model:")
    #Xtform_train = np.c_[X_train[:,0], X_train[:,1], np.ones_like(X_train[:,0])]
    #Xtform_test = np.c_[X_test[:,0], X_test[:,1], np.ones_like(X_test[:,0])]
    Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
            (X_train, np.ones_like(X_train[:,0])),
            (X_test, np.ones_like(X_test[:,0])),
            Y_train,
            Y_test
            )   
    rowtraineval[0,:6], rowtesteval[0,:6] = linreg(Xtform_train, Xtform_test, Ytform_train, Ytform_test)
    
    #Change to polar coordinates
    if(not isPolar):
        print("converting to polar...")
        X_train = get_polar(X_train)
        X_test = get_polar(X_test)
        
    ##DIR DEP
    print("Directional Dependence:")
    #Xtform_train = np.c_[np.cos(X_train[:,1]), np.sin(X_train[:,1]), np.ones_like(X_train[:,1])]
    #Xtform_test = np.c_[np.cos(X_test[:,1]), np.sin(X_test[:,1]), np.ones_like(X_test[:,1])]
    Xtform_train, Xtform_test, Ytform_train, Ytform_test = feature_set(
            (X_train[:,1]),
            (X_test[:,1]),
            Y_train,
            Y_test
            )
    
    rowtraineval[1], rowtesteval[1] = vonmises_dirtune(Xtform_train, Xtform_test, Ytform_train, Ytform_test)
    
    return (node, rowtraineval, rowtesteval)

# %% TUNE DECODING
    
def tune_decode(X, fset, Y, centers, nmods, nmets, mmod='std', savepath='', savename=''):
   
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
    
    # Resize Y so that feature maps are appended as new rows in first feature map
    Y = Y.swapaxes(1,2).reshape((Y.shape[0], Y.shape[2], -1)).swapaxes(1,2)

    """
    if(len(X.shape) > 1):
        x = X[..., centers]
    else:
        x = X
    """
    
    #x = X[:,ikin]
    x = X
    y = Y
    
    ##RESHAPE FOR LINEAR REGRESSION
    #print(x.shape, y.shape)
    
    #if len(x.shape) > 1: ##FOR TIME-BASED DATA ( 2 COMPS PER TIMESPACE IS USE CASE)
    #    tcoff = sum(np.where(centers <= 32, True, False))
    #    x = x[...,32:ntime-32]
   #     y = y[...,tcoff:y.shape[1]-tcoff]
    """
    elif fset == 'labels':
        temp = np.ones_like(y)
        x = (temp.swapaxes(0,1) * x).swapaxes(0,1)
        x = x.reshape((-1,))
    """    
    print(x.shape, y.shape)
    
    #x = x.reshape((-1))
    x = x.swapaxes(1,2).reshape((-1,2))
    #print(x.shape, y.shape)

    y = y.swapaxes(1,2).reshape((-1,y.shape[1]*y.shape[2]))
    #print(x.shape, y.shape)
    

    if(len(x) > 600000):
        nrepeats = 2
        y = np.array([x[320*i+np.random.randint(320)] for i in range(int(len(x)/320))])
        x = np.array([x[320*i+np.random.randint(320)] for i in range(int(len(x)/320))])
        x = np.concatenate((x,y), axis=0)
        y = np.repeat(y, nrepeats, axis=0 )
    else:
        y = np.repeat(y, ntime, axis=0)
    #x = x.reshape((x.shape[0],-1))
    #y = y.reshape((y.shape[0], -1))
    print(x.shape, y.shape)
        
    _, trainevals, testevals, rm = tune_kin_decoding(y,x,0)
    
    #shape evals back to original shape
    #trainevals = trainevals.reshape(unravelshape) #check configuration!!!
    #testevals = testevals.reshape(unravelshape)
    
    if(savepath != '' and savename != ''):
        with open(os.path.join(savepath, '%s_ridgemodel.pkl' %str(savename)),
                'wb') as file:
            rm = pickle.dump(rm, file)
    else:
        print('no file name, cannot save weights')
    
    return trainevals, testevals

# %% TUNE
        
def tune(X, fset, Y, centers, nmods, nmets, mmod='std'):
   
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
        
    pool = mp.Pool(mp.cpu_count())
    
    results = []
    
    # Resize Y so that feature maps are appended as new rows in first feature map
    Y = Y.swapaxes(1,2).reshape((Y.shape[0], Y.shape[2], -1)).swapaxes(1,2)
    
    for irow in range(Y.shape[1]):
        print("Row: ", irow)
        
        #fmapcoord = node[0]
        #fmaptime = node[1]
        #t = centers[fmaptime]
        
        #nodeselector = (slice(None),) + tuple(node)
        
        if(len(X.shape) > 1):
            x = X[..., centers]
        else:
            x = X
        
        y = Y[:,irow]
        
        ##EXCLUDE 10% OF TIME ON EACH SIDE

        
        ##RESHAPE FOR LINEAR REGRESSION
        #print(x.shape, y.shape)
        
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
        #print(x.shape, y.shape)
        
        '''
        if fset == 'labels':
            results.append(pool.apply_async(tune_node_label, args = (x,y,node)))
        elif fset == 'ang':
            results.append(pool.apply_async(tune_node_ang, args=(x,y, node)))
        elif fset == 'angvel':
            results.append(pool.apply_async(tune_node_angvel, args=(x,y,node)))
        elif fset == 'ee':
            results.append(pool.apply_async(tune_node_ee, args=(x,y,node)))
        elif fset == 'eepolar':
            results.append(pool.apply_async(tune_node_eepolar, args=(x,y,node)))
        '''
        
        if mmod == 'std':
            if fset == 'acc':
                results.append(pool.apply_async(tune_row_vel, args=(x,y,irow, True)))
            elif fset == 'vel' or fset == 'eepolar' or fset == 'ang' or fset=='angvel':
                results.append(pool.apply_async(tune_row_vel, args=(x,y,irow, True)))
            elif fset == 'ee':
                results.append(pool.apply_async(tune_row_vel, args=(x,y,irow, False)))
                #sys.stdout.flush()
            elif fset == 'labels':
                results.append(pool.apply_async(tune_row_label, args=(x,y,irow)))
        elif mmod == 'vonmises':
            results.append(pool.apply_async(tune_row_vonmises, args=(x,y,irow)))
            
    results = [r.get() for r in results]
    #print(results)
    
    pool.close()
    print("pool closed")
    pool.join()
    print("pool joined")
    
    for result in results:
        trainevals[result[0]] = result[1]
        testevals[result[0]] = result[2]
    
    #shape evals back to original shape
    trainevals = trainevals.reshape(unravelshape) #check configuration!!!
    testevals = testevals.reshape(unravelshape)
    
    return trainevals, testevals

def tune_layer(X, fset, xyplmvt, runinfo, ilayer, mmod, model, t_stride=2):
    """Wrapper function for tune that performs several layer-wide calculations"""
    
    #savepath = os.path.join(modelname, fset, 'exp%d' %expid)
    #savepath = os.path.join('results', 'exp%d' %expid, base, modelname, fset, runinfo.orientation)
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
    
    lo = pickle.load(open(os.path.join(runinfo.datafolder(model), layer + '.pkl'), 'rb'))
    lo = lo[xyplmvt]
    
    '''
    centers = np.arange(320)
    fmapntime = lo.shape[2]
    #if fmapntime != len(centers):
    for i in range(int(np.log2(len(centers)/ fmapntime))):
        centers = np.array([centers[i*t_stride] for i in range(len(centers)//2)])
    
    assert len(centers) == fmapntime, "length of centers and fmapntime not equal!"
    '''
    centers = get_centers(lo.shape[2])
    
    '''
    if (fset == 'vel' or fset == 'acc'):
        evals = tune_vel(X, lo, centers, mmod)
    #elif(fset == 'acc'):
    #    evals = tune2(X, fset, lo, centers)
    elif(fset == 'angvel'):
        evals = tune(X, fset, lo, centers, 2, 3)
    elif(fset == 'labels'):
        evals = tune(X, fset, lo, centers, 3, 21)
    elif(fset == 'ang'):
        evals = tune(X, fset, lo, centers, 3, 3)
    elif(fset == 'ee' or fset == 'eepolar'):
        evals = tune(X, fset, lo, centers, 4, 3)
    '''
    
    if (fset == 'vel' or fset == 'acc' or fset == 'eepolar' or fset == 'ang' or fset=='angvel' or fset=='ee'):
        nmods = 4
        nmets = 6
    elif (fset == 'labels'):
        nmods = 1
        nmets = 1
    if (mmod == 'vonmises'):
        nmods = 2
        nmets = 7
    
    if (mmod == 'decoding'):
        nmods = 1
        nmets = 3
        trainevals, testevals = tune_decode(X, fset, lo, centers, nmods, nmets, mmod, savepath=savepath, savename=savename)
        #print(savepath, savename)
    else:
        trainevals, testevals = tune(X, fset, lo, centers, nmods, nmets, mmod)
    
    print("Layer %d Completed!" %ilayer)
    
    '''
    maxr2idx = np.dstack(np.unravel_index(np.argsort(- evals[:,1].ravel()), evals.shape))[:5]
    
    print("Highest r2 score values: ", [evals[idx] for idx in maxr2idx])
    print("Node positions: ", maxr2idx)
    '''
    
    np.save('%s/%s_train.npy' %(savepath, savename), trainevals)        
    np.save('%s/%s_test.npy' %(savepath, savename), testevals)
    print("files saved")

# %% DATA READIN
    '''
def get_kinvars(runinfo):
    path_to_data = '../deep_proprioception/dataset/pcr_dataset_test.hdf5'
    kinnames = ['endeffector_coords', 'joint_coords', 'muscle_coords', 'speed']
    with h5py.File(path_to_data, 'r') as datafile:
        idxtups = []
        shape = datafile[kinnames[0]][()].shape
        kinarr = np.zeros((shape[0], 0, shape[2]))
        #with 'speed' as name:
        for name in kinnames:

            kin = datafile[name][()]
            try:
                ncols = datafile[name][()].shape[1]
            except:
                ncols = 1
                kin = kin.reshape(-1,1)
                kin = np.repeat(kin, shape[2], axis=1)
                kin = kin.reshape(kin.shape[0], 1, kin.shape[1])
            idxtups += list(zip([name]*ncols, range(ncols)))
            #speed = datafile[name][()]
            kinarr = np.concatenate((kinarr, kin), axis=1)
            #jointcoords = datafile[name][()]
            
    idx = pd.MultiIndex.from_tuples(idxtups)
    
    np.random.seed(runinfo['randomseed'])
    datafraction = runinfo['datafraction']
    
    if datafraction is not None:
        random_idx = np.random.permutation(data.shape[0])
        subset_num = int(datafraction * random_idx.size)
        data = data[random_idx[:subset_num]]
        labels = labels[random_idx[:subset_num]]
        #jointcoords = jointcoords[random_idx[:subset_num]]
        kinarr = kinarr[random_idx[:subset_num]]
        
    kinvars = pd.Panel(np.swapaxes(kinarr, 0, 1), items=idx)
    
    return kinvars

    '''

    
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
            '''
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            print(a_group_key)
            
            # Get the data
            groups = f[a_group_key]
            
            
            for group in groups:
                print(group)
                print(f[a_group_key][group][:])
            #print(f['data']['block0_values'][:])
            '''
            
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
    #print(ee.shape)
        
    # EXCLUDE SAMPLES THAT AREN'T IN XY PLANE
    #print(runinfo)
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
        #distfrom0 = lambda x : np.linalg.norm(x - x0)
        #X[:,0,:] = np.apply_along_axis(np.linalg.norm, 1, ee)
        #X[:,1,:] = np.apply_along_axis(angle_xaxis, 1, ee)
        X[:,0,:] = np.apply_along_axis(np.linalg.norm, 1, Xrel)
        X[:,1,:] = np.apply_along_axis(angle_xaxis, 1, Xrel)
        #X = np.nan_to_num(X)
    
    elif(fset == 'vel'):
        
        #X[:,0,:] = np.apply_along_axis(np.linalg.norm, 1, vels)
        #X[:,1,:] = np.apply_along_axis(angle_xaxis, 1, vels)
        
        if polar:
            X = get_polar(X)
    
    return X, xyplmvt

# %% MAIN

def main(fset, runinfo, model, startlayer=-1, endlayer=8, mmod='std'):
    
    #assert fset == 'vel' or fset == 'acc' or fset == 'ang' or fset=='labels'\
    #    or fset == 'angvel' or fset == 'ee' or fset == 'eepolar', "Invalid fset!!!"
    assert fset == 'vel' or fset == 'acc' or fset == 'eepolar' or fset == 'ang'\
        or fset == 'labels' or fset=='angvel' or fset=='ee', "Invalid fset!!!"
        
    #assert mmod == '' or mmod == 'flexisin' or mmod == 'rayleigh' or mmod == 'savepd'\
    #    or mmod=='binrayleigh' or mmod=='pycircstat' or mmod=='pdcounts'\
    #    or mmod=='sinbs', 'Invalid mmod!!!'
    assert mmod == 'std' or mmod=='vert' or mmod=='vonmises' or mmod=='decoding', 'Invalid mmod!!!'
    
    #assert mtype == 'S' or mtype =='T' or mtype == 'ST' or mtype=='FC', 'Invalid model type!!!'
    
    #modelname, nlayers = modelinfo(mtype)
    modelname = model['name']
    nlayers = model['nlayers']
    base = model['base']
    
    print('evaluating %s model for %s %s, expid %d, plane %s ...' %(modelname, fset, mmod, runinfo['expid'], runinfo.planestring()))
    
    #print(runinfo.datafolder(model))
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