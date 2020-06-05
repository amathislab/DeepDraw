#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:55:40 2019

@author: kai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:39:39 2019

@author: kai

v5: variable model experiment_id
v4: variable amount of data to save
v2: restructured subfolders
"""

import tensorflow as tf
import yaml, os, h5py
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
#matplotlib.use('default')
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import exp
#import statsmodels.api as sm
from nn_models import ConvModel, AffineModel, RecurrentModel
from nn_train_utils import Dataset
import pickle

# %% SETUP
#PARS
#datafraction = 1/40

#modelname = 'gru_2_32-32_256_low'
#modelname = 'lstm_4_8-32-64-64_512_med' #test
#modelname = 'lstm_4_32-32-64-64_256_low'
#modelname = 'fc_4_8-16-32-64_8-8-32-64_73_low'
#modelname = 'spatiotemporal_4_8-8-32-64_7292'
#modelname = 'temporal_spatial_4_16-16-32-64_64-64-64-64_5272'
#modelname = 'spatial_temporal_4_8-16-16-32_64-64-64-64_5272'
#kinnames = ['end_id', 'endeffector_coords', 'joint_coords', 'label', 'muscle_coords', \
#           'plane', 'rot', 'shear_x', 'shear_y', 'size', 'speed', 'spindle_firing', \
#           'start_id', 'startpt']

kinnames = ['endeffector_coords', 'joint_coords', 'muscle_coords', 'speed']
#kinnames = ['joint_coords', 'muscle_coords']
#twinsize = 5


# %% DATASET PREP
#KINETIC DATA

def main(modelinfo, runinfo):
    
    #print(modelname)
    modelname = modelinfo['name']
    modelbase = modelinfo['base']
    datafraction = runinfo['datafraction']
    
    np.random.seed(runinfo['randomseed'])
    
    print(modelname)

    #PATHS
    #model_path = f"/home/kai/Dropbox/deep_proprioception/models/experiment_3/{modelname}/"
    #model_path = f"/home/kai/Desktop/deep_proprioception/models/experiment_3/{modelname}/"
    model_path = f"models/experiment_{runinfo.model_experiment_id}/{modelname}/"
    path_to_data = '../deep_proprioception/dataset/pcr_dataset_test.hdf5'
    PATH_TO_DATA = '../deep_proprioception/dataset/'
    #MODELS_DIR = '/home/kai/Desktop/deep_proprioception/models/'
    MODELS_DIR = '.'
    #path_to_config_file = f"/home/kai/Desktop/deep_proprioception/models/experiment_3/{modelname}/config.yaml"
    path_to_config_file = f"models/experiment_{runinfo.model_experiment_id}/{modelname}/config.yaml"
    
    if path_to_data is not None:
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
            
            """
            ee = datafile['endeffector_coords'][()]
            firing = datafile['spindle_firing'][()]
            jointcoords = datafile['joint_coords'][()]
            musclecoords = datafile['muscle_coords'][()]
            rot = datafile['rot'][()]
            labels = datafile['label'][()] - 1
            """
    
    idx = pd.MultiIndex.from_tuples(idxtups)
    
    #SPINDLE FIRING TEST DATA
            
    test_data_path = os.path.join(PATH_TO_DATA, 'pcr_dataset_test.hdf5')
    dataset = Dataset(test_data_path, dataset_type='test')
    
    #Extract needed data
    data = dataset.test_data
    labels = dataset.test_labels
    
    # For when I want to use only a fraction of the dataset to train!
    if datafraction is not None:
        random_idx = np.random.permutation(data.shape[0])
        subset_num = int(datafraction * random_idx.size)
        data = data[random_idx[:subset_num]]
        labels = labels[random_idx[:subset_num]]
        #jointcoords = jointcoords[random_idx[:subset_num]]
        kinarr = kinarr[random_idx[:subset_num]]
    
    nsamples, ninputs, ntime = data.shape
    #batch_size = 1
    batch_size = nsamples
    num_steps = nsamples // batch_size
    
    # CREATE PANDAS PANEL
    kinvars = pd.Panel(np.swapaxes(kinarr, 0, 1), items=idx)
    
    # INITIALIZE MODEL
    tf.reset_default_graph()
    
    """
    # import the graph from the file
    imported_graph = tf.train.import_meta_graph(model_path + "model.ckpt.meta")
    
    # list all the tensors in the graph
    for tensor in tf.get_default_graph().get_operations():
        print (tensor.name)
    """
    
    with open(path_to_config_file, 'r') as myfile:
        model_config = yaml.load(myfile)
        train_mean = model_config['train_mean']

    #model_config['model_path'] = 
    model = ConvModel(model_config['experiment_id'], model_config['nclasses'], model_config['arch_type'], \
                      int(model_config['nlayers']), model_config['n_skernels'], model_config['n_tkernels'], \
                      int(model_config['s_kernelsize']), int(model_config['t_kernelsize']), int(model_config['s_stride']), 
                      int(model_config['t_stride']))
    
    '''
    model = AffineModel(
                experiment_id=model_config['experiment_id'],
                nclasses=model_config['nclasses'],
                nlayers=int(model_config['nlayers']),
                nunits=model_config['nunits'],
                nkernels=model_config['nkernels'],
                kernelsize=int(model_config['kernelsize']),
                stride=int(model_config['stride']),
                keep_prob=model_config['keep_prob'])
    '''
    
    '''
    model = RecurrentModel(
        experiment_id=model_config['experiment_id'],
        nclasses=model_config['nclasses'],
        npplayers=int(model_config['npplayers']),
        nppunits=model_config['nppunits'],
        rec_blocktype='gru',
        n_recunits=int(model_config['n_recunits']),
        keep_prob=model_config['keep_prob'])
    '''
    
    # RUN PREDICTIONS AND SAVE INFORMATION FOR TUNING CURVE
    mygraph = tf.Graph()
    with mygraph.as_default():
        
        ##BUILD GRAPH
        # Declare placeholders for input data and labels
        X = tf.placeholder(tf.float32, shape=[batch_size, ninputs, ntime], name="X")
        y = tf.placeholder(tf.int32, shape=[batch_size], name="y")
    
        # Compute scores and accuracy
        scores, probabilities, net = model.predict(X, is_training=False)
        correct = tf.nn.in_top_k(probabilities, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")    
        
        # Build layer output reader
        #layers = [np.zeros(layer.get_shape().as_list()) for layer in list(net.values())]
    
        """
        # Test the `model`!
        restorer = tf.train.Saver()
        myconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        with tf.Session(config=myconfig) as sess:
            ckpt_filepath = os.path.join(model.model_path, 'model.ckpt')
            restorer.restore(sess, ckpt_filepath)
            
            test_accuracy = []
            for step in range(num_steps):         
                batch_x, batch_y = dataset.next_valbatch(batch_size, 'test', step)
                acc = sess.run([accuracy], feed_dict={X: batch_x - train_mean, y: batch_y})
                test_accuracy.append(acc)
        """   
        
        """
        Results of net for this model:
            OrderedDict([('spatial0',
                  <tf.Tensor 'Network/Spatial0/Relu:0' shape=(1, 25, 320, 8) dtype=float32>),
                 ('spatial1',
                  <tf.Tensor 'Network/Spatial1/Relu:0' shape=(1, 25, 320, 8) dtype=float32>),
                 ('temporal0',
                  <tf.Tensor 'Network/Temporal0/Relu:0' shape=(1, 25, 320, 8) dtype=float32>),
                 ('temporal1',
                  <tf.Tensor 'Network/Temporal1/Relu:0' shape=(1, 25, 320, 32) dtype=float32>),
                 ('score',
                  <tf.Tensor 'Network/Classifier/BiasAdd:0' shape=(1, 20) dtype=float32>)])
        """
             
        ##PROBE RECEPTIVE FIELDS
        
        if(not modelinfo['control']):
            #print('added %s' %modelname[-2:])
            model.model_path = model.model_path + modelname[-2:] #Add control set number
        else:
            model.model_path = model.model_path + modelname[-3:]
        restorer = tf.train.Saver()
        myconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        with tf.Session(config=myconfig) as sess:
            ckpt_filepath = os.path.join(model.model_path, 'model.ckpt')
            restorer.restore(sess, ckpt_filepath)
            layers = sess.run(list((net.values())), feed_dict={X: data, y: labels})
            #print(s0, t0, o0)
    
    #SAVE FOLLOW THROUGH
    
    datafolder = runinfo.datafolder(modelinfo)
    os.makedirs(datafolder, exist_ok=True)
    #os.mkdir(datafolder)
    kinvars.to_hdf(datafolder + "/kinvars.hdf5", key="data")
    print("Kinvars saved")
    
    pickle.dump(data, open(datafolder + "/data.pkl", "wb"))
    print("MF saved")
    
    pickle.dump(labels, open(datafolder + "/labels.pkl", "wb"))
    print("Labels saved")
    
    for i in range(len(layers) - 1):
        pickle.dump(layers[i], open(datafolder + f"/l{i}.pkl", "wb"))
        print(f"L{i} saved")
        