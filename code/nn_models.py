'''
Class and forward pass definitions for various neural network models.

'''


import os
from collections import OrderedDict

import tensorflow as tf
slim = tf.contrib.slim
cudnn_rnn = tf.contrib.cudnn_rnn

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(CUR_DIR), 'nn-training/')


class ConvModel():
    """Defines a convolutional neural network model of the proprioceptive system."""

    def __init__(self,
            experiment_id, nclasses, arch_type, nlayers, n_skernels, n_tkernels, s_kernelsize,
            t_kernelsize, s_stride, t_stride, seed=None, train=True):
        """Set up hyperparameters of the convolutional network.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        arch_type : {'spatial_temporal', 'spatiotemporal', 'temporal_spatial'} str, defines the type
            of convolutional neural network model.
        nlayers : int, number of layers in the cnn model.
        n_skernels : list of ints, number of kernels for spatial processing.
        n_tkernels : list of ints, number of kernels for temporal processing.
        s_kernelsize : int, size of the spatial kernel.
        t_kernelsize : int, size of the temporal kernel.
        s_stride : int, stride along the spatial dimension.
        t_stride : int, stride along the temporal dimension.
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
        self.nclasses = nclasses
        self.arch_type = arch_type
        self.nlayers = nlayers
        self.n_tkernels = n_tkernels
        self.n_skernels = n_skernels
        self.t_kernelsize = t_kernelsize
        self.s_kernelsize = s_kernelsize
        self.t_stride = t_stride
        self.s_stride = s_stride
        self.seed = seed

        # Make model name
        if arch_type == 'spatial_temporal':
            kernels = ('-'.join(str(i) for i in n_skernels)) + '_' + ('-'.join(str(i) for i in n_tkernels))
        elif arch_type == 'temporal_spatial':
            kernels = ('-'.join(str(i) for i in n_tkernels)) + '_' + ('-'.join(str(i) for i in n_skernels))
        else:
            kernels = ('-'.join(str(i) for i in n_skernels))

        parts_name = [arch_type, str(nlayers), kernels,
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
        X : tf.tensor [batch_size, num_inputs, num_timesteps], input tensor for which scores must
            be calculated.

        Returns
        -------
        score : tf.tensor [batch_size, nclasses], computed scores by passing X through the network.
        probabilities : tf.tensor [batch_size, nclasses], softmax probabilities.
        net : orderedDict, contains all layer representations.

        """
        net = OrderedDict([])

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            score = tf.expand_dims(X, -1)
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

                score = tf.reshape(score, [batch_size, -1])
                score = slim.dropout(score, 0.7, is_training=is_training)
                score = slim.fully_connected(score, self.nclasses, None, scope='Classifier')

                net['score'] = score

                probabilities = tf.nn.softmax(score, name="Y_proba")

        return score, probabilities, net


class AffineModel():
    """Defines a neural network model of the proprioceptive system where the spatial
    processing happens through fully connected layers and temporal processing through
    convolutional layers.

    """

    def __init__(
            self, experiment_id, nclasses, nlayers, nunits, nkernels, kernelsize, stride, keep_prob):
        """Set up the hyperparameters of the affine model.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        nlayers : int, number of layers in the affine model.
        nunits : list of ints, number of units in the affine layers for spatial processing.
        nkernels : list of ints, number of kernels for temporal processing.
        kernelsize : int, size of the temporal kernel.
        stride : int, stride along the temporal dimension.
        keep_prob : float, amount of dropout at each spatial processing layer.

        """
        assert (len(nkernels) == nlayers), \
            "Number of spatial and temporal processing layers must be equal!"

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.nlayers = nlayers
        self.nunits = nunits
        self.nkernels = nkernels
        self.kernelsize = kernelsize
        self.stride = stride
        self.keep_prob = keep_prob

        # Make model name
        dropout_name = {'0.6': 'high', '0.7': 'med', '0.8': 'low'}
        kernels = ('-'.join(str(i) for i in nunits)) + '_' + ('-'.join(str(i) for i in nkernels))
        parts_name = ['fc', str(nlayers), kernels, (str(kernelsize) + str(stride)), dropout_name[str(keep_prob)]]

        # Create model directory
        self.name = '_'.join(parts_name)
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
            score = tf.transpose(X, [0, 2, 1])
            batch_size = X.get_shape()[0]

            fully_connected = lambda score, layer_id: slim.fully_connected(
                score, self.nunits[layer_id], normalizer_fn=slim.layer_norm, scope=f'FC{layer_id}')
            temporal_conv = lambda score, layer_id: slim.conv2d(
                score, self.nkernels[layer_id], [self.kernelsize, 1], [self.stride, 1],
                data_format='NHWC', normalizer_fn=slim.layer_norm, scope=f'Conv{layer_id}')

            for layer in range(self.nlayers):
                score = fully_connected(score, layer)
                score = slim.dropout(score, keep_prob=self.keep_prob, is_training=is_training)
                net[f'spatial{layer}'] = score

            score = tf.expand_dims(score, -1)

            for layer in range(self.nlayers):
                score = temporal_conv(score, layer)
                net[f'temporal{layer}'] = score

            score = tf.reshape(score, [batch_size, -1])
            score = slim.dropout(score, keep_prob=0.7, is_training=is_training)
            score = slim.fully_connected(score, self.nclasses, activation_fn=None, scope='Classifier')

            net['score'] = score

            probabilities = tf.nn.softmax(score, name="Y_proba")

        return score, probabilities, net


class RecurrentModel():
    """Defines a recurrent neural network model of the proprioceptive system."""

    def __init__(
            self, experiment_id, nclasses, rec_blocktype, n_recunits, npplayers, nppunits, keep_prob):
        """Set up the hyperparameters of the recurrent model.

        Arguments
        ---------
        experiment_id : int, identifier for model path
        nclasses : int, number of classes in the classification problem.
        rec_blocktype: {'lstm', 'gru'} str, type of recurrent block.
        n_recunits : int, number of units in the recurrent block.
        npplayers : int, number of layers in the fully-connected module.
        nppunits : list of ints, number of units in the affine layers for spatial processing.
        keep_prob : float, amount of dropout at each spatial processing layer.

        """

        assert len(nppunits) == npplayers
        assert rec_blocktype in ('lstm', 'gru')

        self.experiment_id = experiment_id
        self.nclasses = nclasses
        self.rec_blocktype = rec_blocktype
        self.n_recunits = n_recunits
        self.npplayers = npplayers
        self.nppunits = nppunits
        self.keep_prob = keep_prob

        # Make model name
        dropout_name = {'0.6': 'high', '0.7': 'med', '0.8': 'low'}
        units = ('-'.join(str(i) for i in nppunits))
        parts_name = [rec_blocktype, str(npplayers), units, str(n_recunits), dropout_name[str(keep_prob)]]

        # Create model directory
        self.name = '_'.join(parts_name)
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
            score = tf.transpose(X, [0, 2, 1])
            batch_size = X.get_shape()[0]

            fully_connected = lambda score, layer_id: slim.fully_connected(
                score, self.nppunits[layer_id], normalizer_fn=slim.layer_norm, scope=f'FC{layer_id}')

            for layer in range(self.npplayers):
                score = fully_connected(score, layer)
                score = slim.dropout(score, keep_prob=self.keep_prob, is_training=is_training)
                net[f'spatial{layer}'] = score

            # `cudnn_rnn` requires the inputs to be of shape [timesteps, batch_size, num_inputs]
            score = tf.transpose(score, [1, 0, 2])
            if self.rec_blocktype == 'lstm':
                recurrent_cell = cudnn_rnn.CudnnLSTM(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)
            elif self.rec_blocktype == 'gru':
                recurrent_cell = cudnn_rnn.CudnnGRU(1, self.n_recunits, name='RecurrentBlock')
                score, _ = recurrent_cell.apply(score)

            score = tf.transpose(score, [1, 0, 2])
            net['recurrent_out'] = score

            score = slim.dropout(score, 0.7, is_training=is_training)
            score = slim.fully_connected(score, self.nclasses, activation_fn=None, scope='Classifier')

            net['score'] = score

            probabilities = tf.nn.softmax(score[:, -1, :], name="Y_proba")

        return score[:, -1, :], probabilities, net
