import os
import random
import argparse
import time

import pandas
import pickle

import sys
sys.path.append('../code')
from nn_models import ConvModel
from nn_train_utils import *

PATH_TO_DATA = '../dataset'


def main(args):
    # Load dataset
    train_data_path = os.path.join(PATH_TO_DATA, 'pcr_dataset_train.hdf5')
    train_data = Dataset(train_data_path, 'train')
    
    all_conv_models = pickle.load(open('./convmodels.p', 'rb'))
    
    model_number = args.id
    arch_type = args.type
    
    best_models_arch = all_conv_models[all_conv_models['arch_type'] == arch_type].nlargest(1, 'test_accuracy')
    latents = best_models_arch.iloc[model_number]

    seeds = [1, 2, 3, 4, 5]
    
    for train_type in [False, True]:        
        for myseed in seeds:
            # Create model
            mymodel = ConvModel(
                experiment_id=4,
                nclasses=20,
                arch_type=arch_type,
                nlayers=latents['nlayers'],
                n_skernels=latents['n_skernels'],
                n_tkernels=latents['n_tkernels'],
                s_kernelsize=latents['s_kernelsize'],
                t_kernelsize=latents['t_kernelsize'],
                s_stride=latents['s_stride'],
                t_stride=latents['t_stride'],
                seed=myseed,
                train=train_type)

            print(mymodel.__dict__)

            # Create trainer and train!
            mytrainer = Trainer(mymodel, train_data)
            mytrainer.train(num_epochs=50, verbose=False, save_rand=not(train_type))

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training Convolutional Nets for PCR.')
    parser.add_argument('--type', type=str, help='Type of ConvNet')
    parser.add_argument('--id', type=int, help='Id of net to train')
    main(parser.parse_args())