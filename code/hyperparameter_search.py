import os
import random
import argparse
import time
import numpy as np
import yaml

from nn_models import ConvModel, AffineModel, RecurrentModel
from nn_train_utils import *

PATH_TO_DATA = '/gpfs01/bethge/home/pmamidanna/deep_proprioception/data/'

def sample_affine_model():
    n_layers = [1, 2, 3, 4]
    n_kernels = [8, 16, 32, 64]
    kernel_size = [3, 5, 7, 9]
    stride = [1, 2, 3]
    dropout_frac = [0.8, 0.7, 0.6]

    # need to split and arrange the bloody kernels and kernelsizes
    layers = random.choice(n_layers)

    # Create model
    mymodel = AffineModel(
        experiment_id=4,
        nclasses=20,
        nlayers=layers,
        nunits=sorted(random.choices(n_kernels, k=layers)),
        nkernels=sorted(random.choices(n_kernels, k=layers)),
        kernelsize=random.choice(kernel_size),
        stride=random.choice(stride),
        keep_prob=random.choice(dropout_frac))
    
    return mymodel

def sample_recurrent_model(archtype):
    n_layers = [1, 2, 3, 4]
    n_units = [8, 16, 32, 64]
    n_recunits = [256, 512]
    dropout_frac = [0.8, 0.7, 0.6]

    layers = random.choice(n_layers)

    mymodel = RecurrentModel(
        experiment_id=4,
        nclasses=20,
        npplayers=layers,
        nppunits=sorted(random.choices(n_units, k=layers)),
        rec_blocktype=archtype,
        n_recunits=random.choice(n_recunits),
        keep_prob=random.choice(dropout_frac))

    return mymodel

def main(args):
    # Load dataset
    train_data_path = os.path.join(PATH_TO_DATA, 'pcr_dataset_train.hdf5')
    train_data = Dataset(train_data_path, 'train')

    learning_rates = 10**np.random.uniform(-6, -2, size=15)
    batch_sizes = [128, 256]

    num_models = 3

    for i in range(num_models):  
        if args.type == 'FC':
            mymodel = sample_affine_model()
        elif args.type == 'Recurrent':
            mymodel = sample_recurrent_model(args.archtype)
        display(mymodel.__dict__)

        # Create trainer and train!
        for j in range(5):
            intime = time.time()
            mymodel.model_path += f'_{j}'
            mytrainer = Trainer(mymodel, train_data)
            mylearning_rate = random.choice(learning_rates)
            mybatch_size = random.choice(batch_sizes)
            mytrainer.train(num_epochs=10, early_stopping_steps=1000, verbose=False, 
                learning_rate=mylearning_rate, batch_size=mybatch_size)
            outt = time.time()

            mydict = {'learning_rate': float(mylearning_rate),
                      'batch_size': int(mybatch_size),
                      'time_to_train': float(outt-intime)}
            path_to_yaml_file = os.path.join(mymodel.model_path, 'config.yaml')
            with open(path_to_yaml_file, 'a') as myfile:
                yaml.dump(mydict, myfile, default_flow_style=False)

            print(f'Successfully trained model {i+1} / {num_models} in {(outt-intime)/60} minutes.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter search for FC/Recurrents nets.')
    parser.add_argument('--type', type=str, help='Type of network {FC, Recurrent}')
    parser.add_argument('--archtype', type=str, default=None, help='Subtype of network. {GRU, LSTM}')
    main(parser.parse_args())