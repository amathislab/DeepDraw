import os
import sys
from nn_models import ConvModel, RecurrentModel, AffineModel

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
import h5py
# from scipy.stats import mode
# from scipy.spatial.distance import pdist, squareform


def load_model(meta_data, experiment_id, model_type, is_trained):
    '''Load a trained `ConvModel`, `AffineModel` or `RecurrentModel` object.

    Returns
    -------
    model : an instance of `ConvModel`, `AffineModel` or `RecurrentModel`

    '''
    if model_type == 'conv':
        
        try:
            myseed = meta_data['seed']
        except:
            myseed = None
        
        model = ConvModel(
            experiment_id=experiment_id,
            nclasses=20,
            arch_type=meta_data['arch_type'],
            nlayers=meta_data['nlayers'],
            n_skernels=meta_data['n_skernels'],
            n_tkernels=meta_data['n_tkernels'],
            s_kernelsize=int(meta_data['s_kernelsize']),
            t_kernelsize=int(meta_data['t_kernelsize']),
            s_stride=int(meta_data['s_stride']),
            t_stride=int(meta_data['t_stride']),
            seed=myseed,
            train=is_trained)

    return model


def set_layer_dimensions(representations, model_type, layer_name):
    '''Set dimensions of the layer representations to (n_examples, n_timesteps, n_dims).

    Arguments
    ---------
    representations : np.array, whose dimensions must be changed
    model_type : str {conv, affine, recurrent}
    layer_name : str, name of the layer retrieved from basename of representations file

    '''
    if model_type == 'recurrent':
        return representations
    elif model_type == 'affine':
        if 'temporal' in layer_name:
            n_e, n_t, _, _ = representations.shape
            return representations.reshape((n_e, n_t, -1))
        else:
            return representations
    elif model_type == 'conv':
        n_e, n_s, n_t, n_c = representations.shape
        representations = representations.transpose((0, 2, 1, 3))

    return representations.reshape((n_e, n_t, n_s*n_c))


def set_kin_dimensions(kinematics, n_out_timesteps):
    '''Set dimensions of the kinematic variables to be decoded to (n_examples, n_out_timesteps, n_dims).

    Arguments
    ---------
    kinematics : np.array (n_examples, n_dims, n_timesteps)
    n_out_timesteps : int, number of timesteps in the layer representation

    '''
    n_examples, n_dims, n_timesteps = kinematics.shape
    assert n_timesteps == 320, "Kinematics shape mismatch. Please revise."
    skip_idx = n_timesteps // (n_out_timesteps - 1)
    kinematics = kinematics[:, :, ::skip_idx]
    assert kinematics.shape[-1] == n_out_timesteps, "Whaatt!"
    return kinematics.transpose(0, 2, 1)


def get_representations(h5filepath):
    '''Retrieve generated representations for the queried layer.

    Returns
    -------
    representations: np.array of shape (num_examples, num_timesteps, num_inputs)

    '''
    reps_layer = h5py.File(h5filepath, 'r')
    num_datasets = len(list(reps_layer))
    ds_shape = list(reps_layer.get('0').shape)

    batch_size = ds_shape[0]
    ds_shape[0] = batch_size*num_datasets
    representations = np.zeros((ds_shape))
    for i in range(num_datasets):
        representations[batch_size*i : batch_size*(i+1)] = reps_layer.get(str(i))[()]

    return representations


def generate_layerwise_representations(model, data, batch_size, repfile_name):
    '''Generate layerwise representations and save in separate hdf5 files.

    Arguments
    ---------
    model : instance of `ConvModel`, `AffineModel` or `RecurrentModel`.
    data : np.array of spindle inputs for which representations are to be generated.
    batch_size : int, batch size to use for generating representations.
    repfile_name : str, prefix for retrieving representations easily using the glob module.

    '''

    def make_h5file_name(model_path, repfile_name, layername):
        h5file_name = repfile_name + '_' + layername + '.hdf5'
        return os.path.join(model_path, h5file_name)

    # Retrieve training mean, if data was normalized
    path_to_config_file = os.path.join(model.model_path, 'config.yaml')
    with open(path_to_config_file, 'r') as myfile:
        model_config = yaml.load(myfile)
    
    try:
        train_mean = model_config['train_mean']
    except:
        train_mean = 0

    num_examples = data.shape[0]
    myshape = list(data.shape)
    myshape[0] = batch_size

    h5files_dict = {}
    num_batches = num_examples // batch_size

    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, myshape, name="X")
        _, _, output = model.predict(X, is_training=False)
        restorer = tf.train.Saver()
        myconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

        with tf.Session(config=myconfig) as sess:
            ckpt_filepath = os.path.join(model.model_path, 'model.ckpt')
            restorer.restore(sess, ckpt_filepath)

            for i in range(num_batches):
                out = sess.run(output, feed_dict={X: data[batch_size*i : batch_size*(i+1)] - train_mean})
                for key in out.keys():
                    if i == 0:
                        h5files_dict[key] = h5py.File(
                            make_h5file_name(model.model_path, repfile_name, key), 'w')
                    h5files_dict[key].create_dataset(str(i), data=out[key])

    for key, val in h5files_dict.items():
        val.close()


def compute_rdm(representations):
    """Given a bunch of representations, calculates correlations for each pair of representations.
    
    Returns
    -------
    rdm : np.ndarray of shape (num_traj, num_traj), num_traj is number of trajectories that make 
        up the dataset.
    
    """
    
    num_traj = representations.shape[0]
    flat_reps = np.reshape(representations, (num_traj, -1))
    
    # Compute correlations
    rdm = pdist(flat_reps, metric='correlation')
    
    return rdm