import os
import random
import argparse
import time

import pandas
import pickle

import sys
sys.path.append('../code')
from nn_models import ConvModel, RecurrentModel
from nn_train_utils import *

PATH_TO_DATA = '../../dataset/'

def sample_conv_latents(arch_type):
    n_layers = [1, 2, 3, 4]
    n_kernels = [8, 16, 32, 64]
    kernel_size = [3, 5, 7, 9]
    s_stride = [1, 2]
    t_stride = [1, 2, 3]

    # need to split and arrange the kernels and kernelsizes
    layers = random.choice(n_layers)
    kernels = sorted(random.choices(n_kernels, k=2*layers))
    kernelsizes = sorted(random.choices(kernel_size, k=2))

    latents = {
        'layers': layers,  
        's_kernelsize': kernelsizes[0], 
        't_kernelsize': kernelsizes[1],
        's_stride': random.choice(s_stride), 
        't_stride': random.choice(t_stride)}

    if arch_type == 'spatial_temporal':
        latents['n_skernels'] = kernels[:layers]
        latents['n_tkernels'] = kernels[layers:]
    elif arch_type == 'temporal_spatial':
        latents['n_skernels'] = kernels[layers:]
        latents['n_tkernels'] = kernels[:layers]
    elif arch_type == 'spatiotemporal':
        kernels = sorted(random.choices(n_kernels, k=layers))
        latents['n_skernels'] = kernels
        latents['n_tkernels'] = kernels

    return latents

def sample_rec_latents():
    npplayers = [1, 2, 3]
    nppfilters = [8, 16, 32]
    n_recunits = [32, 64, 128]
    s_kernelsize = [3, 5, 7, 9]
    s_stride = [1, 2]
    
    layers = random.choice(npplayers)

    latents = {'npplayers': layers,
        'nppfilters': sorted(random.choices(nppfilters, k=layers)),
        'n_recunits': random.choice(n_recunits),
        's_kernelsize': random.choice(s_kernelsize),
        's_stride': random.choice(s_stride)}

    return latents

def main(args):
    # Load dataset
    train_data_path = os.path.join(PATH_TO_DATA, 'pcr_dataset_train.hdf5')
    train_data = Dataset(train_data_path, 'train', key='spindle_info')

    num_models = args.number
    model_type = args.type
    arch_type = args.arch_type

    for i in range(num_models):
        # Sample model hyperparameters
        if model_type == 'conv':
            latents = sample_conv_latents(arch_type)
            mymodel = ConvModel(
                experiment_id=args.exp_id,
                nclasses=20,
                arch_type=arch_type,
                nlayers=latents['layers'],
                n_skernels=latents['n_skernels'],
                n_tkernels=latents['n_tkernels'],
                s_kernelsize=latents['s_kernelsize'],
                t_kernelsize=latents['t_kernelsize'],
                s_stride=latents['s_stride'],
                t_stride=latents['t_stride'])

        elif model_type == 'rec':
            latents = sample_rec_latents()
            mymodel = RecurrentModel(
                experiment_id=args.exp_id,
                nclasses=20,
                rec_blocktype=arch_type,
                n_recunits=latents['n_recunits'],
                npplayers=latents['npplayers'],
                nppfilters=latents['nppfilters'],
                s_kernelsize=latents['s_kernelsize'],
                s_stride=latents['s_stride'])

        intime = time.time()
        print(mymodel.__dict__)

        # Create trainer and train!
        mytrainer = Trainer(mymodel, train_data)
        if model_type == 'rec':
            mytrainer.train(num_epochs=50, learning_rate=1e-3, batch_size=128, 
            early_stopping_epochs=5, verbose=False)
        else:
            mytrainer.train(num_epochs=50, verbose=False)
        outt = time.time()
        print(f'Successfully trained model {i+1} / {num_models} in {(outt-intime)/60} minutes.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training Convolutional Nets for PCR.')
    parser.add_argument('--type', type=str, help='Type of Neural Net')
    parser.add_argument('--exp_id', type=str, help='Experiment ID')
    parser.add_argument('--arch_type', type=str, help='Sub-Type of architecture')
    parser.add_argument('--number', type=int, help='Number of nets to train')
    main(parser.parse_args())
