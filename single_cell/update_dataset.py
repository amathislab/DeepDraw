#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:27:03 2020

@author: kai
"""

import tensorflow as tf
import yaml, os, h5py
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import exp
from nn_models import ConvModel, AffineModel, RecurrentModel
from nn_train_utils import Dataset
import pickle

basefolder = '/mnt/data/revisions/analysis-data/' 

with h5py.File(basefolder + '../pcr_data/pcr_dataset_test.hdf5', 'r') as file:
    muscle_inputs_test = file['muscle_coords'][()]
vel_inputs_test = np.gradient(muscle_inputs_test, 0.015, axis=2)
spindle_info_test = np.stack((muscle_inputs_test, vel_inputs_test), axis=-1)
with h5py.File(basefolder + '../pcr_data/pcr_dataset_test.hdf5', 'a') as file:
    file.create_dataset('spindle_info', data=spindle_info_test)
    
print('dataset updated. this does not need to be run again')