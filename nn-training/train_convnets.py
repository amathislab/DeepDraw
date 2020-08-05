import os
import random
import argparse
import time

from nn_models import ConvModel, AffineModel, RecurrentModel
from nn_train_utils import *

PATH_TO_DATA = '../dataset/'

def sample_latents(arch_type):
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

def main(args):
    # Load dataset
    train_data_path = os.path.join(PATH_TO_DATA, 'pcr_dataset_train.hdf5')
    train_data = Dataset(train_data_path, 'train')

    num_models = args.number
    arch_type = args.type

    for i in range(num_models):
        # Sample model hyperparameters
        latents = sample_latents(arch_type)
        intime = time.time()

        # Create model
        mymodel = ConvModel(
            experiment_id=1,
            nclasses=20,
            arch_type=arch_type,
            nlayers=latents['layers'],
            n_skernels=latents['n_skernels'],
            n_tkernels=latents['n_tkernels'],
            s_kernelsize=latents['s_kernelsize'],
            t_kernelsize=latents['t_kernelsize'],
            s_stride=latents['s_stride'],
            t_stride=latents['t_stride'])

        display(mymodel.__dict__)

        # Create trainer and train!
        mytrainer = Trainer(mymodel, train_data)
        mytrainer.train(num_epochs=50, verbose=False)
        outt = time.time()
        print(f'Successfully trained model {i+1} / {num_models} in {(outt-intime)/60} minutes.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training Convolutional Nets for PCR.')
    parser.add_argument('--type', type=str, help='Type of ConvNet')
    parser.add_argument('--number', type=int, help='Number of nets to train')
    main(parser.parse_args())