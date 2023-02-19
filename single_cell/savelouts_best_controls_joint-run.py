#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:39:39 2019

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
import pickle#, time
from tensorflow.contrib.rnn import *

# %% SETUP
kinnames = ['endeffector_coords', 'joint_coords', 'muscle_coords', 'speed']

# %% DATASET PREP
#KINETIC DATA

def main(modelinfo, runinfo):
    ''' Save the model hidden layer activations
    
    Parameters
    ----------
    modelinfo : dict, model information
    runinfo : RunInfo, experimental run information
    '''
    
    modelname = modelinfo['name']
    modelbase = modelinfo['base']
    datafraction = runinfo['datafraction']
    
    np.random.seed(runinfo['randomseed'])
    
    print(modelname)

    #PATHS
    basefolder = runinfo['basefolder']
    
    model_path = f"{basefolder}models/experiment_{runinfo.model_experiment_id}/{modelname}/"
    #path_to_data = '../deep_proprioception/dataset/pcr_dataset_test.hdf5'
    #PATH_TO_DATA = '../deep_proprioception/dataset/'
    path_to_data = f'{basefolder}../pcr_data/pcr_dataset_test.hdf5'
    PATH_TO_DATA = f'{basefolder}../pcr_data/'
    MODELS_DIR = '.'
    path_to_config_file = f"{basefolder}models/experiment_{runinfo.model_experiment_id}/{modelname}/config.yaml"
    
    if path_to_data is not None:
        with h5py.File(path_to_data, 'r') as datafile:
            idxtups = []
            shape = datafile[kinnames[0]][()].shape
            kinarr = np.zeros((shape[0], 0, shape[2]))
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
                kinarr = np.concatenate((kinarr, kin), axis=1)
    
    idx = pd.MultiIndex.from_tuples(idxtups)
    
    #SPINDLE FIRING TEST DATA
    test_data_path = os.path.join(PATH_TO_DATA, 'pcr_dataset_test.hdf5')
    dataset = Dataset(test_data_path, dataset_type='test', key='spindle_info')    
 
    #Extract needed data
    data = dataset.test_data
    labels = dataset.test_labels
    
    # For when I want to use only a fraction of the dataset to train!
    if datafraction is not None:
        random_idx = np.random.permutation(data.shape[0])
        subset_num = int(datafraction * random_idx.size)
        data = data[random_idx[:subset_num]]
        labels = labels[random_idx[:subset_num]]
        kinarr = kinarr[random_idx[:subset_num]]
    
    nsamples, ninputs, ntime, _ = data.shape
    #batch_size = nsamples
    batch_size = 100 #can be updated based on GPU capacities for forward pass
    #batch_size = 25 #can be updated based on GPU capacities for forward pass
    num_steps = nsamples // batch_size
    
    # CREATE PANDAS PANEL
    print('kinarr shape', kinarr.shape)
    kinvars = pd.Panel(np.swapaxes(kinarr, 0, 1), items=idx)
    #time.sleep(10)
    
    # INITIALIZE MODEL
    tf.reset_default_graph()
    
    with open(path_to_config_file, 'r') as myfile:
        model_config = yaml.load(myfile)
        train_mean = model_config['train_mean']


    if (modelinfo['type'] in ['S', 'ST']):
        model = ConvModel(model_config['experiment_id'], model_config['nclasses'], model_config['arch_type'], \
                          int(model_config['nlayers']), model_config['n_skernels'], model_config['n_tkernels'], \
                          int(model_config['s_kernelsize']), int(model_config['t_kernelsize']), int(model_config['s_stride']), 
                          int(model_config['t_stride']))
    
    else:        
        print('building rec model')
        model = RecurrentModel(model_config['experiment_id'], model_config['nclasses'], model_config['rec_blocktype'], 
                               int(model_config['n_recunits']), int(model_config['npplayers']), list(map(int, model_config['nppfilters'])), 
                               int(model_config['s_kernelsize']), int(model_config['s_stride']))
    
    model.model_path = basefolder + model.model_path
    
    layers = []
    
    mygraph = tf.Graph()
    with mygraph.as_default():
        # Declare placeholders for input data and labels
        X = tf.placeholder(tf.float32, shape=[batch_size, ninputs, ntime, 2], name="X")
        y = tf.placeholder(tf.int32, shape=[batch_size], name="y")

        # Compute scores and accuracy
        scores, probabilities, net = model.predict(X, is_training=False)
        correct = tf.nn.in_top_k(probabilities, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

        # Test the `model`!
        restorer = tf.train.Saver()
        myconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        
        if(not modelinfo['control']):
            model.model_path = model.model_path + modelname[-2:] #Add control set number
        else:
            model.model_path = model.model_path + modelname[-3:]

        #update model path
        print('model.model_path', model.model_path)
        
        """
        with tf.Session(config=myconfig) as sess:
            ckpt_filepath = os.path.join(model.model_path, 'model.ckpt')
            print('checkpoint filepath', ckpt_filepath)
            restorer.restore(sess, ckpt_filepath)
            
            for i in range(num_steps):
                if(runinfo.verbose >= 1):
                    print('batch %d / %d' %(i, num_steps))
                layers_batch = sess.run(list((net.values())), \
                        feed_dict={X: data[batch_size*i:batch_size*(i+1)], y: labels[batch_size*i:batch_size*(i+1)]})
                
                for j in range(len(layers_batch) - 1):
                    if(i == 0):
                        #layers.append(layers_batch[j])
                        layers.append(np.zeros(tuple([nsamples]) + layers_batch[j].shape[1:]))
                    #else:
                        #print(layers_batch[j].shape, layers[j].shape)
                    #    layers[j] = np.concatenate((layers[j], layers_batch[j]))
                        #print(layers[j].shape)
                    layers[j][i*batch_size : (i+1)*batch_size] = layers_batch[j]
        """
        with tf.Session(config=myconfig) as sess:
            ckpt_filepath = os.path.join(model.model_path, 'model.ckpt')
            print('checkpoint filepath', ckpt_filepath)
            restorer.restore(sess, ckpt_filepath)
            
            for i in range(num_steps):
                if(runinfo.verbose >= 1):
                    print('batch %d / %d' %(i, num_steps))
                layers_batch = sess.run(list((net.values())), \
                        feed_dict={X: data[batch_size*i:batch_size*(i+1)], y: labels[batch_size*i:batch_size*(i+1)]})
                
                for j in range(len(layers_batch) - 1):
                    if(i == 0):
                        #layers.append(layers_batch[j])
                        layers.append(np.zeros(tuple([nsamples]) + layers_batch[j].shape[1:]))
                    #else:
                        #print(layers_batch[j].shape, layers[j].shape)
                    #    layers[j] = np.concatenate((layers[j], layers_batch[j]))
                        #print(layers[j].shape)
                    layers[j][i*batch_size : (i+1)*batch_size] = layers_batch[j]
            
    #SAVE FOLLOW THROUGH    
    datafolder = runinfo.datafolder(modelinfo)
    os.makedirs(datafolder, exist_ok=True)
    kinvars.to_hdf(datafolder + "/kinvars.hdf5", key="data")
    print("Kinvars saved")
    
    pickle.dump(data, open(datafolder + "/data.pkl", "wb"))
    print("MF saved")
    
    pickle.dump(labels, open(datafolder + "/labels.pkl", "wb"))
    print("Labels saved")
    
    for i in range(len(layers)):
        pickle.dump(layers[i], open(datafolder + f"/l{i}.pkl", "wb"))
        print(f"L{i} saved")    