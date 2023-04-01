'''
Class and forward pass definitions for various neural network models.

These functions/modules are mostly taken from the training code elsewhere in the repository.

'''


import os
from collections import OrderedDict

import tensorflow as tf
slim = tf.contrib.slim
cudnn_rnn = tf.contrib.cudnn_rnn

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(CUR_DIR), '../nn-training/')


class ConvRModel():
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(self,
            experiment_id, arch_type, nlayers, n_skernels, n_tkernels, s_kernelsize,
            t_kernelsize, s_stride, t_stride, noutspace=3, seed=None, train=True):
        """Set up hyperparameters of the convolutional network.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        arch_type : {'spatial_temporal', 'spatiotemporal', 'temporal_spatial'} str, defines the type
            of convolutional neural network model.
        nlayers : int, number of layers in the cnn model.
        n_skernels : list of ints, number of kernels for spatial processing.
        n_tkernels : list of ints, number of kernels for temporal processing.
        s_kernelsize : int, size of the spatial kernel.
        t_kernelsize : int, size of the temporal kernel.
        s_stride : int, stride along the spatial dimension.
        t_stride : int, stride along the temporal dimension.
        noutspace: int, number of spatial dimensions of the kinematic variable that we are decoding
        seed : int, for saving random initializations of networks.
        train : bool, is the network meant to be trained or not

        """

        assert (len(n_skernels) == len(n_tkernels) == nlayers), \
            "Number of spatial and temporal processing layers must be equal!"
        if arch_type == 'spatiotemporal':
            n_tkernels = n_skernels
            t_kernelsize = s_kernelsize
            t_stride = s_stride

        self.experiment_id = experiment_id
        self.arch_type = arch_type
        self.nlayers = nlayers
        self.n_tkernels = n_tkernels
        self.n_skernels = n_skernels
        self.t_kernelsize = t_kernelsize
        self.s_kernelsize = s_kernelsize
        self.t_stride = t_stride
        self.s_stride = s_stride
        self.seed = seed
        self.noutspace = noutspace

        # Make model name
        if arch_type == 'spatial_temporal':
            kernels = ('-'.join(str(i) for i in n_skernels)) + '_' + ('-'.join(str(i) for i in n_tkernels))
        elif arch_type == 'temporal_spatial':
            kernels = ('-'.join(str(i) for i in n_tkernels)) + '_' + ('-'.join(str(i) for i in n_skernels))
        else:
            kernels = ('-'.join(str(i) for i in n_skernels))

        parts_name = [arch_type, 'r', str(nlayers), kernels,
                      ''.join(str(i) for i in [s_kernelsize, s_stride, t_kernelsize, t_stride])]

        # Create model directory
        self.name = '_'.join(parts_name)
        
        if seed is not None: self.name += '_' + str(self.seed)
        if not train: self.name += 'r'
            
        exp_dir = os.path.join(MODELS_DIR, f'experiment_{self.experiment_id}')
        self.model_path = os.path.join(exp_dir, self.name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

    def predict(self, X, is_training=True):
        """Computes the scores (forward pass) for the given network.

        Arguments
        ---------
        X : tf.tensor [batch_size, num_inputs, num_timesteps, 2], input tensor for which scores must
            be calculated.

        Returns
        -------
        score : tf.tensor [batch_size, nclasses], computed scores by passing X through the network.
        probabilities : tf.tensor [batch_size, nclasses], softmax probabilities.
        net : orderedDict, contains all layer representations.

        """
        net = OrderedDict([])

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]

            with slim.arg_scope([slim.conv2d], data_format='NHWC', normalizer_fn=slim.layer_norm):

                spatial_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_skernels[layer_id], [self.s_kernelsize, 1], [self.s_stride, 1],
                    scope=f'Spatial{layer_id}')
                temporal_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_tkernels[layer_id], [1, self.t_kernelsize], [1, self.t_stride],
                    scope=f'Temporal{layer_id}')
                spatiotemporal_conv = lambda score, layer_id: slim.conv2d(
                    score, self.n_skernels[layer_id], [self.s_kernelsize, self.t_kernelsize], 
                    [self.s_stride, self.t_stride], scope=f'Spatiotemporal{layer_id}')

                if self.arch_type == 'spatial_temporal':
                    for layer in range(self.nlayers):
                        score = spatial_conv(score, layer)
                        net[f'spatial{layer}'] = score
                    for layer in range(self.nlayers):
                        score = temporal_conv(score, layer)
                        net[f'temporal{layer}'] = score

                elif self.arch_type == 'temporal_spatial':
                    for layer in range(self.nlayers):
                        score = temporal_conv(score, layer)
                        net[f'temporal{layer}'] = score
                    for layer in range(self.nlayers):
                        score = spatial_conv(score, layer)
                        net[f'spatial{layer}'] = score

                elif self.arch_type == 'spatiotemporal':
                    for layer in range(self.nlayers):
                        score = spatiotemporal_conv(score, layer)
                        net[f'spatiotemporal{layer}'] = score
            
            outtime = score.get_shape()[2]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, outtime, -1])
            score = slim.fully_connected(score, self.noutspace, activation_fn=None, scope='Classifier')
            
            net['score'] = score

        return score, net


class RecurrentRModel():
    """Defines a recurrent neural network model of the proprioceptive system."""

    def __init__(
            self, experiment_id, rec_blocktype, n_recunits, npplayers, nppfilters, 
            s_kernelsize, s_stride, noutspace=3, seed=None, train=True):
        """Set up the hyperparameters of the recurrent model.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        rec_blocktype: {'lstm', 'gru'} str, type of recurrent block.
        n_recunits : int, number of units in the recurrent block.
        npplayers : int, number of layers in the fully-connected module.
        nppfilters : list of ints, number of filters (spatial convolutions) for spatial processing.
        s_kernelsize : int, size of conv kernel
        s_stride : int, stride for conv kernel
        noutspace: int, number of spatial dimensions of the kinematic variable we want to decode
        seed : int, for saving random initializations
        train : bool, whether to train the model or not (just save random initialization)

        """

        assert len(nppfilters) == npplayers
        assert rec_blocktype in  ('lstm', 'gru')

        self.experiment_id = experiment_id
        self.rec_blocktype = rec_blocktype
        self.n_recunits = n_recunits
        self.npplayers = npplayers
        self.nppfilters = nppfilters
        self.s_kernelsize = s_kernelsize
        self.s_stride = s_stride
        self.seed = seed
        self.noutspace = noutspace

        # Make model name
        units = ('-'.join(str(i) for i in nppfilters))
        parts_name = [rec_blocktype, 'r', str(npplayers), units, str(n_recunits)]

        # Create model directory
        self.name = '_'.join(parts_name)
        if seed is not None: self.name += '_' + str(self.seed)
        if not train: self.name += 'r'

        exp_dir = os.path.join(MODELS_DIR, f'experiment_{self.experiment_id}')
        self.model_path = os.path.join(exp_dir, self.name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Additional useful parameters
        self.num_parameters = 0
        self.is_training = True

    def predict(self, X, is_training=True):

        net = OrderedDict()

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = X
            batch_size = X.get_shape()[0]

            spatial_conv = lambda score, layer_id: slim.conv2d(
                score, self.nppfilters[layer_id], [self.s_kernelsize, 1], [self.s_stride, 1],
                data_format='NHWC', normalizer_fn=slim.layer_norm, scope=f'Spatial{layer_id}')

            for layer in range(self.npplayers):
                score = spatial_conv(score, layer)
                net[f'spatial{layer}'] = score

            # `cudnn_rnn` requires the inputs to be of shape [timesteps, batch_size, num_inputs]
            score = tf.transpose(score, [0, 2, 1, 3])
            score = tf.reshape(score, [batch_size, 320, -1])
            score = tf.transpose(score, [1, 0, 2])
            
            if self.rec_blocktype == 'lstm':
                recurrent_cell = cudnn_rnn.CudnnLSTM(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'gru':
                recurrent_cell = cudnn_rnn.CudnnGRU(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)

            score = tf.transpose(score, [1, 0, 2])
            net['recurrent_out'] = score
            
            score = slim.fully_connected(score, self.noutspace, activation_fn=None, scope='Classifier')
            net['score'] = score

        return score, net
