"""
"""

import os
import copy
import h5py
import yaml

import numpy as np
import tensorflow as tf


class Dataset():
    """Defines a dataset object with simple routines to generate batches."""

    def __init__(self, path_to_data=None, data=None, dataset_type='train', key='spindle_firing', fraction=None):
        """Set up the `Dataset` object.
        Arguments
        ---------
        path_to_data : str, absolute location of the dataset file.
        dataset_type : {'train', 'test'} str, type of data that will be used along with the model.
        key : {'endeffector_coords', 'joint_coords', 'muscle_coords', 'spindle_firing'} str
        """
        self.path_to_data = path_to_data
        self.dataset_type = dataset_type
        self.key = key
        self.train_data = self.train_labels = None
        self.val_data = self.val_labels = None
        self.test_data = self.test_data = None
        self.make_data(data)

        # For when I want to use only a fraction of the dataset to train!
        if fraction is not None:
            random_idx = np.random.permutation(self.train_data.shape[0])
            subset_num = int(fraction * random_idx.size)
            self.train_data = self.train_data[random_idx[:subset_num]]
            self.train_labels = self.train_labels[random_idx[:subset_num]]

    def make_data(self, mydata):
        """Load train/val or test splits into the `Dataset` instance.
        Returns
        -------
        if dataset_type == 'train' : loads train and val splits.
        if dataset_type == 'test' : loads the test split.
        """
        # Load and shuffle dataset randomly before splitting
        if self.path_to_data is not None:
            with h5py.File(self.path_to_data, 'r') as datafile:
                data = datafile[self.key][()]
                labels = datafile['label'][()] - 1
        else: 
            data = mydata['data']
            labels = mydata['labels'] - 1

        # For training data, create training and validation splits
        if self.dataset_type == 'train':
            self.train_data, self.train_labels, self.val_data, self.val_labels = train_val_split(
                data, labels)

        # For test data, do not split
        elif self.dataset_type == 'test':
            self.test_data, self.test_labels = data, labels

    def next_trainbatch(self, batch_size, step=0):
        """Returns a new batch of training data.
        Arguments
        ---------
        batch_size : int, size of training batch.
        step : int, step index in the epoch.
        Returns
        -------
        2-tuple of batch of training data and correspondig labels.
        """
        if step == 0:
            shuffle_idx = np.random.permutation(self.train_data.shape[0])
            self.train_data = self.train_data[shuffle_idx]
            self.train_labels = self.train_labels[shuffle_idx]
        mybatch_data = self.train_data[batch_size*step:batch_size*(step+1)]
        mybatch_labels = self.train_labels[batch_size*step:batch_size*(step+1)]

        return (mybatch_data, mybatch_labels)

    def next_valbatch(self, batch_size, type='val', step=0):
        """Returns a new batch of validation or test data.
        Arguments
        ---------
        type : {'val', 'test'} str, type of data to return.
        """
        if type == 'val':
            mybatch_data = self.val_data[batch_size*step:batch_size*(step+1)]
            mybatch_labels = self.val_labels[batch_size*step:batch_size*(step+1)]
        elif type == 'test':
            mybatch_data = self.test_data[batch_size*step:batch_size*(step+1)]
            mybatch_labels = self.test_labels[batch_size*step:batch_size*(step+1)]

        return (mybatch_data, mybatch_labels)


class Trainer:
    """Trains a `Model` object with the given `Dataset` object."""

    def __init__(self, model=None, dataset=None, global_step=None):
        """Set up the `Trainer`.
        Arguments
        ---------
        model : an instance of `ConvModel`, `AffineModel` or `RecurrentModel` to be trained.
        dataset : an instance of `Dataset`, containing the train/val data splits.
        """
        self.model = model
        self.dataset = dataset
        self.log_dir = model.model_path
        self.global_step = 0 if global_step == None else global_step
        self.session = None
        self.graph = None
        self.best_loss = 1e10
        self.validation_accuracy = 0

    def build_graph(self):
        """Build training graph using the `Model`s predict function and setting up an optimizer."""
        
        _, ninputs, ntime, _ = self.dataset.train_data.shape
        with tf.Graph().as_default() as self.graph:
            tf.set_random_seed(self.model.seed)
            # Placeholders
            self.learning_rate = tf.placeholder(tf.float32)
            self.X = tf.placeholder(tf.float32, shape=[self.batch_size, ninputs, ntime, 2], name="X")
            self.y = tf.placeholder(tf.int32, shape=[self.batch_size], name="y")

            # Set up optimizer, compute and apply gradients
            scores, probabilities, _ = self.model.predict(self.X, is_training=True)
            classification_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=scores)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(classification_loss)

            # Calculate metrics
            scores_eval, prob_eval, _ = self.model.predict(self.X, is_training=False)
            self.val_loss_op = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=scores_eval)
            correct = tf.nn.in_top_k(prob_eval, self.y, 1)
            self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

            tf.summary.scalar('Train_Loss', classification_loss)
            tf.summary.scalar('Train_Acc', self.accuracy_op)
            self.train_summary_op = tf.summary.merge_all()
            
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def load(self):
        self.saver.restore(self.session, os.path.join(self.log_dir, 'model.ckpt'))
        
    def save(self):
        self.saver.save(self.session, os.path.join(self.log_dir, 'model.ckpt'))

    def train(self,
            num_epochs=10,
            learning_rate=0.0005,
            batch_size=256,
            val_steps=100,
            early_stopping_epochs=5,
            retrain=False,
            normalize=False,
            verbose=True, 
            save_rand=False):
        """Train the `Model` object.
        Arguments
        ---------
        num_epochs : int, number of epochs to train for.
        learning_rate : float, learning rate for Adam Optimizer.
        batch_size : int, size of batch to train on.
        val_steps : int, number of batches after which to perform validation.
        early_stopping_steps : int, number of steps for early stopping criterion.
        retrain : bool, train already existing model vs not.
        normalize : bool, whether to normalize training data or not.
        verbose : bool, print progress on screen.
        """
        steps_per_epoch = self.dataset.train_data.shape[0] // batch_size
        max_iter = num_epochs * steps_per_epoch
        early_stopping_steps = early_stopping_epochs * steps_per_epoch
        self.batch_size = batch_size

        if normalize:
            self.train_data_mean = float(np.mean(self.dataset.train_data))
            self.train_data_std = float(np.std(self.dataset.train_data))
        else:
            self.train_data_mean = self.train_data_std = 0
        train_params = {'train_mean': self.train_data_mean,
                        'train_std': self.train_data_std}
        val_params = {'validation_loss': 1e10, 'validation_accuracy': 0}

        self.build_graph()
        self.session = tf.Session(graph=self.graph)

        # with self.graph.as_default():
        self.session.run(self.init)
        if retrain:
            self.load()
        
        if save_rand:
            self.save()
            self.model.is_training = False
            make_config_file(self.model, train_params, val_params)
            self.session.close()
            return

        # Create summaries
        self.train_summary = tf.summary.FileWriter(
            os.path.join(self.model.model_path, 'train'), graph=self.graph, flush_secs=30)
        self.val_summary = tf.summary.FileWriter(os.path.join(self.model.model_path, 'val'))

        not_improved = 0
        end_training = 0
        val_params = {}

        for self.global_step in range(max_iter):
            
            # Training step
            batch_X, batch_y = self.dataset.next_trainbatch(
                batch_size, self.global_step % steps_per_epoch)
            feed_dict = {self.X: batch_X - self.train_data_mean,
                        self.y: batch_y, 
                        self.learning_rate: learning_rate}
            self.session.run(self.train_op, feed_dict)

            # Validate/save periodically
            if self.global_step % val_steps == 0:
                # Summarize, print progress
                loss_val, acc_val = self.save_summary(feed_dict)
                if verbose:
                    print('Step : %4d, Validation Accuracy : %.2f' % (self.global_step, acc_val))

                if loss_val < self.best_loss:
                    self.best_loss = loss_val
                    self.validation_accuracy = acc_val
                    self.save()
                    val_params = {
                        'validation_loss': float(self.best_loss), 
                        'validation_accuracy': float(acc_val)}
                    not_improved = 0
                else:
                    not_improved += 1

                if not_improved >= early_stopping_steps:
                    learning_rate /= 4
                    print(learning_rate)
                    not_improved = 0
                    end_training += 1
                    self.load()

                if end_training == 2:
                    if self.global_step < 40*steps_per_epoch:
                        end_training = 1
                        not_improved = 0
                    else:
                        break

        self.model.is_training = False
        make_config_file(self.model, train_params, val_params)
        self.session.close()

    def save_summary(self, feed_dict):
        """Create and save summaries for training and validation."""
        train_summary = self.session.run(self.train_summary_op, feed_dict)
        self.train_summary.add_summary(train_summary, self.global_step)
        validation_loss, validation_accuracy = self.eval()
        validation_summary = tf.Summary()
        validation_summary.value.add(tag='Validation_Loss', simple_value=validation_loss)
        validation_summary.value.add(tag='Validation_Acc', simple_value=validation_accuracy)
        self.val_summary.add_summary(validation_summary, self.global_step)
        
        return validation_loss, validation_accuracy

    def eval(self):
        """Evaluate validation performance.
        
        Returns
        -------
        validation_loss : float, loss on the entire validation data
        validation_accuracy : float, accuracy on the validation data
        
        """
        num_iter = self.dataset.val_data.shape[0] // self.batch_size
        acc_val = np.zeros(num_iter)
        loss_val = np.zeros(num_iter)
        for i in range(num_iter):
            batch_X, batch_y = self.dataset.next_valbatch(self.batch_size, step=i)
            feed_dict = {self.X: batch_X - self.train_data_mean, self.y: batch_y}
            loss_val[i], acc_val[i] = self.session.run([self.val_loss_op, self.accuracy_op], feed_dict)
        return loss_val.mean(), acc_val.mean()


def evaluate_model(model, dataset, batch_size=200):
    """Evaluation routine for trained models.
    Arguments
    ---------
    model : the `Conv`, `Affine` or `Recurrent` model to be evaluated. The test data is 
        assumed to be defined within the model.dataset object.
    dataset : the `Dataset` object on which the model is to be evaluated.
    Returns
    -------
    accuracy : float, Classification accuracy of the model on the given dataset.
    """
    # Data handling
    nsamples, ninputs, ntime, _ = dataset.test_data.shape
    num_steps = nsamples // batch_size

    # Retrieve training mean, if data was normalized
    path_to_config_file = os.path.join(model.model_path, 'config.yaml')
    with open(path_to_config_file, 'r') as myfile:
        model_config = yaml.load(myfile)
    train_mean = model_config['train_mean']

    mygraph = tf.Graph()
    with mygraph.as_default():
        # Declare placeholders for input data and labels
        X = tf.placeholder(tf.float32, shape=[batch_size, ninputs, ntime, 2], name="X")
        y = tf.placeholder(tf.int32, shape=[batch_size], name="y")

        # Compute scores and accuracy
        scores, probabilities, _ = model.predict(X, is_training=False)
        correct = tf.nn.in_top_k(probabilities, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

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

    return np.mean(test_accuracy)


# Auxiliary Functions

def train_val_split(data, labels):
    num_train = int(0.9*data.shape[0])
    train_data, train_labels = data[:num_train], labels[:num_train]
    val_data, val_labels = data[num_train:], labels[num_train:]

    return (train_data, train_labels, val_data, val_labels)


def make_config_file(model, train_params, val_params):
    """Make a configuration file for the given model, created after training.
    Given a `ConvModel`, `AffineModel` or `RecurrentModel` instance, generates a 
    yaml file to save the configuration of the model.
    """
    mydict = copy.copy(model.__dict__)
    # Convert to python native types for better readability
    for (key, value) in mydict.items():
        if isinstance(value, np.generic):
            mydict[key] = float(value)
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            mydict[key] = [int(item) for item in value]

    # Save yaml file in the model's path
    path_to_yaml_file = os.path.join(model.model_path, 'config.yaml')
    with open(path_to_yaml_file, 'w') as myfile:
        yaml.dump(mydict, myfile, default_flow_style=False)
        yaml.dump(train_params, myfile, default_flow_style=False)
        yaml.dump(val_params, myfile, default_flow_style=False)

    return