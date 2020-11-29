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


def main(args):
    # Load dataset
    train_data_path = os.path.join(PATH_TO_DATA, 'pcr_dataset_train.hdf5')
    train_data = Dataset(train_data_path, 'train', key='spindle_info')
    
    if args.type == 'conv':
        all_models = pickle.load(open('./newconvmodels.p', 'rb'))
        best_model_arch = all_models[all_models['arch_type'] == args.arch_type].nlargest(1, 'test_accuracy')
    elif args.type == 'rec':
        all_models = pickle.load(open('./recmodels.p', 'rb'))
        best_model_arch = all_models[all_models['rec_blocktype'] == args.arch_type].nlargest(1, 'test_accuracy')
    latents = best_model_arch.iloc[0]

    seeds = [1, 2, 3, 4, 5]
    
    for train_type in [False, True]:        
        for myseed in seeds:
            # Create model
            if args.type == 'conv':
                mymodel = ConvModel(
                    experiment_id=args.exp_id,
                    nclasses=20,
                    arch_type=args.arch_type,
                    nlayers=latents['nlayers'],
                    n_skernels=latents['n_skernels'],
                    n_tkernels=latents['n_tkernels'],
                    s_kernelsize=latents['s_kernelsize'],
                    t_kernelsize=latents['t_kernelsize'],
                    s_stride=latents['s_stride'],
                    t_stride=latents['t_stride'],
                    seed=myseed,
                    train=train_type)
            elif args.type == 'rec':
                mymodel = RecurrentModel(
                    experiment_id=args.exp_id,
                    nclasses=20,
                    rec_blocktype=args.arch_type,
                    n_recunits=latents['n_recunits'],
                    npplayers=latents['npplayers'],
                    nppfilters=latents['nppfilters'],
                    s_kernelsize=latents['s_kernelsize'],
                    s_stride=latents['s_stride'],
                    seed=myseed,
                    train=train_type)

            print(mymodel.__dict__)

            # Create trainer and train!
            mytrainer = Trainer(mymodel, train_data)
            if args.type == 'conv':
                mytrainer.train(num_epochs=50, verbose=False, save_rand=not(train_type))
            elif args.type == 'rec':
                mytrainer.train(num_epochs=50, learning_rate=1e-3, batch_size=128, 
                    early_stopping_epochs=5, verbose=False, save_rand=not(train_type))

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training Best Net initializations for PCR.')
    parser.add_argument('--type', type=str, help='Type of Network')
    parser.add_argument('--arch_type', type=str, help='Subtype')
    parser.add_argument('--exp_id', type=int, help='Experment_id to SAVE model')
    main(parser.parse_args())