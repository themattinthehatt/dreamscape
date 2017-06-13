#!/usr/bin/env python

"""Variational Autoencoder class

@author: Matt Whiteway, March 2017
VAE class implements a variational autoencoder

"""

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import tensorflow as tf


class VAE(object):
    """Variational Autoencoder class
    
    Attributes:
        layers_encoder (list of ints): size of each layer in encoder, excluding
            input layer
        input_size (int): size of input
        layer_latent (int): size of latent layer
        layers_decoder (list of ints): size of each layer in decoder, including
            output layer
        num_lvs (int): size of latent layer
        num_layers_enc (int): number of layers in encoder, including input 
            layer
        num_layers_dec (int): number of layers in decoder, including latent
            layer and output layer
        act_func (str): activation function for network layers
        
        weights_enc (list of tf.Variable): weights of the encoding network
        biases_enc (list of tf.Variable): biases of the encoding network
        weights_mean (tf.Variable): weights from encoding network to latent 
            variable distribution mean
        biases_mean (tf.Variable): biases from encoding network to latent 
            variable distribution mean
        weights_log_var (tf.Variable): weights from encoding network to log of
            latent variable distribution variance
        biases_log_var (tf.Variable): biases from encoding network to log of
            latent variable distribution variance
        weights_dec (list of tf.Variable): weights of the decoding network
        biases_dec (list of tf.Variable): biases of the decoding network
        
        x (tf placeholder): ph for input to model
        z_mean (tf op): mean value for each latent variable
        z_log_var (tf op): log of the variance for each latent variable
        z (tf op): sample value of latent variable
        eps (tf placeholder): ph for N(0,1) input to stochastic layer
        x_recon (tf op): reconstructed input values
        cost (tf op): evaluates the cost function of the network
        
        learning_rate (float): global learning rate used by gradient descent 
            optimizers
        train_step (tf op): evaluates one training step using the specified 
            cost function and learning algorithm
        
        graph (tf.Graph): dataflow graph for the network
        saver (tf.train.Saver): for saving and restoring variables
        merge_summaries (tf op): op that merges all summary ops
        init (tf op): op that initializes global variables in graph 
    
    """

    def __init__(
            self,
            layers_encoder=None,
            layer_latent=None,
            layers_decoder=None,
            act_func='relu',
            learning_rate=1e-3):
        """Constructor for VAE class
        
        Args:
            layers_encoder (list of ints): size of each layer in encoder, 
                including input layer
            layer_latent (int): size of latent layer
            layers_decoder (list of ints): size of each layer in decoder, 
                including output layer
            act_func (str): activation function for network layers
                ['relu'] | 'sigmoid' | 'tanh' | 'linear' | 'softplus' | 'elu'
            learning_rate (scalar): global learning rate for gradient descent 
                methods
            
        Raises:
            InputError if layers_encoder is not specified
            InputError if layers_latent is not specified
            InputError if layers_decoder is not specified
            InputError if act_func is not a valid string
            
        """

        # input checking
        if layers_encoder is None:
            raise InputError('Must specify layer sizes for encoder')
        if layer_latent is None:
            raise InputError('Must specify number of latent dimensions')
        if layers_decoder is None:
            raise InputError('Must specify layer sizes for decoder')

        self.input_size = layers_encoder[0]
        self.layers_encoder = layers_encoder[1:]
        self.layer_latent = layer_latent
        self.layers_decoder = layers_decoder

        if act_func == 'relu':
            self.act_func = tf.nn.relu
        elif act_func == 'sigmoid':
            self.act_func = tf.sigmoid
        elif act_func == 'tanh':
            self.act_func = tf.tanh
        elif act_func == 'linear':
            self.act_func = tf.identity
        elif act_func == 'softplus':
            self.act_func = tf.nn.softplus
        elif act_func == 'elu':
            self.act_func = tf.nn.elu
        else:
            raise InputError('Invalid activation function')

        self.learning_rate = learning_rate

        # define useful constants
        self.num_lvs = self.layer_latent
        self.num_layers_enc = len(self.layers_encoder)
        self.num_layers_dec = len(self.layers_decoder)

        # for saving and restoring models
        self.graph = tf.Graph()  # must be initialized before graph creation

        # build model graph
        with self.graph.as_default():

            # define pipeline for feeding data into model
            with tf.variable_scope('data'):
                self._initialize_data_pipeline()

            # initialize weights and create encoder model
            with tf.variable_scope('encoder'):
                self._define_recognition_network()

            # initialize weights and create decoder model
            with tf.variable_scope('decoder'):
                self._define_generator_network()

            # define loss function
            with tf.variable_scope('loss'):
                self._define_loss()

            # define optimizer
            with tf.variable_scope('optimizer'):
                self._define_optimizer()

            # add additional ops
            # for saving and restoring models
            self.saver = tf.train.Saver()  # must be init after var creation
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()
            # self.init = tf.initialize_all_variables()

    def _initialize_data_pipeline(self):
        """Create placeholders for input and random values"""

        self.x = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.input_size],
            name='input_ph')
        self.eps = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.num_lvs],
            name='rand_ph')

    def _define_recognition_network(self):
        """ 
        Create a recognition network to transform inputs into its latent 
        representation
        """

        # push data through the encoding function to determine mean and std
        # of latent vars
        self.weights_enc = []
        self.biases_enc = []
        z_enc = [self.x]
        for layer in range(self.num_layers_enc):
            with tf.variable_scope(str('layer_%01i' % layer)):

                # initialize weights
                if layer == 0:
                    in_size = self.input_size
                else:
                    in_size = self.layers_encoder[layer - 1]
                out_size = self.layers_encoder[layer]
                self.weights_enc.append(tf.get_variable(
                    shape=[in_size, out_size],
                    name=str('weights_enc_%01i' % layer),
                    initializer=tf.random_normal_initializer(stddev=0.1)))

                # initialize biases
                self.biases_enc.append(tf.get_variable(
                    initializer=tf.zeros(shape=[1, out_size]),
                    name=str('biases_enc_%01i' % layer)))

                # calculate layer activations
                pre = tf.add(
                    tf.matmul(z_enc[layer], self.weights_enc[layer]),
                    self.biases_enc[layer])
                post = self.act_func(pre)
                z_enc.append(post)

                # save summaries of layer activations
                with tf.variable_scope('summaries'):
                    tf.summary.histogram('pre_act', pre)
                    tf.summary.histogram('post_act', post)

        with tf.variable_scope('latent_layer'):

            # initialize weights/biases for means of stochastic layer
            self.weights_mean = tf.get_variable(
                shape=[self.layers_encoder[-1], self.num_lvs],
                name='weights_mean',
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.biases_mean = tf.get_variable(
                initializer=tf.zeros(shape=[1, self.num_lvs]),
                name='biases_mean')

            # initialize weights/biases for log variances of stochastic layer
            self.weights_log_var = tf.get_variable(
                shape=[self.layers_encoder[-1], self.num_lvs],
                name='weights_log_var',
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.biases_log_var = tf.get_variable(
                initializer=tf.zeros(shape=[1, self.num_lvs]),
                name='biases_log_var')

            # weights to estimate mean of normally distributed latent vars
            self.z_mean = tf.add(
                tf.matmul(z_enc[-1], self.weights_mean), self.biases_mean,
                name='z_mean')
            # estimating log of the variance is easier since the latent loss
            # has a log determinant term
            self.z_log_var = tf.add(
                tf.matmul(z_enc[-1], self.weights_log_var),
                self.biases_log_var,
                name='z_log_var')

            # transform estimated mean and log variance into a sampled value
            # of the latent state using z = mu + sigma*epsilon
            self.z = tf.add(
                self.z_mean,
                tf.multiply(tf.sqrt(tf.exp(self.z_log_var)), self.eps))

            # save summaries of means and log_vars
            with tf.variable_scope('summaries'):
                tf.summary.histogram('means', self.z_mean)
                tf.summary.histogram('log_vars', self.z_log_var)

    def _define_generator_network(self):
        """ 
        Create a generator network to transform a random sample
        in the latent space into an image
        """

        self.weights_dec = []
        self.biases_dec = []
        z_dec = [self.z]
        for layer in range(self.num_layers_dec):
            with tf.variable_scope(str('layer_%01i' % layer)):

                # initialize weights
                if layer == 0:
                    in_size = self.num_lvs
                else:
                    in_size = self.layers_decoder[layer - 1]
                out_size = self.layers_decoder[layer]
                self.weights_dec.append(tf.get_variable(
                    shape=[in_size, out_size],
                    name=str('weights_dec_%01i' % layer),
                    initializer=tf.truncated_normal_initializer(stddev=0.1)))

                # initialize biases
                self.biases_dec.append(tf.get_variable(
                    initializer=tf.zeros(shape=[1, out_size]),
                    name=str('biases_dec_%01i' % layer)))

                # calculate layer activations
                pre = tf.add(
                    tf.matmul(z_dec[layer], self.weights_dec[layer]),
                    self.biases_dec[layer])
                post = self.act_func(pre)
                z_dec.append(post)

                # define this for easier access later
                self.x_recon = z_dec[-1]

                # save summaries of layer activations
                with tf.variable_scope('summaries'):
                    tf.summary.histogram('pre_act', pre)
                    tf.summary.histogram('post_act', post)

    def _define_loss(self):
        """Define loss function that will be used to optimize model params"""

        # define reconstruction loss
        loss_recon = 0.5 * tf.reduce_sum(tf.square(self.x_recon - self.x), 1)

        # define latent loss
        loss_latent = 0.5 * tf.reduce_sum(tf.exp(self.z_log_var)
                                          + tf.square(self.z_mean)
                                          - 1 - self.z_log_var, 1)

        # define cost
        self.cost = tf.reduce_mean(loss_recon + loss_latent)

    def _define_optimizer(self):
        """Define one step of the optimization routine"""
        self.train_step = tf.train.AdamOptimizer(self.learning_rate). \
            minimize(self.cost)

    def train(
            self,
            sess,
            data=None,
            batch_size=128,
            epochs_training=10,
            epochs_disp=None,
            epochs_ckpt=None,
            epochs_summary=None,
            output_dir=None):
        """Network training
        
        Args:
            sess (tf.Session object): current session object to run graph
            data (DataReader object): input to network
            batch_size (int, optional): batch size used by the gradient
                descent-based optimizers
            epochs_training (int, optional): number of epochs for gradient 
                descent-based optimizers
            epochs_disp (int, optional): number of epochs between updates to 
                the console
            epochs_ckpt (int, optional): number of epochs between saving 
                checkpoint files
            epochs_summary (int, optional): number of epochs between saving
                network summary information 
            output_dir (string, optional): absolute path for saving checkpoint
                files and summary files; must be present if either epochs_ckpt  
                or epochs_summary is not 'None'.

        Returns:
            None

        Raises:
            InputError: If epochs_ckpt is not None and output_dir is None
            InputError: If epochs_summary is not None and output_dir is None
            
        """

        # check input
        if data is None:
            raise InputError('data reader must be specified')
        if epochs_ckpt is not None and output_dir is None:
            raise InputError('output_dir must be specified to save model')
        if epochs_summary is not None and output_dir is None:
            raise InputError('output_dir must be specified to save summaries')

        # initialize file writers
        if epochs_summary is not None:
            test_writer = tf.summary.FileWriter(
                os.path.join(output_dir, 'summaries', 'test'),
                sess.graph)

        # begin training
        with self.graph.as_default():

            num_batches = int(data.train.num_examples / batch_size)

            # start training loop
            for epoch in range(epochs_training):

                for batch in range(num_batches):

                    # get batch of data for this training step
                    x = data.train.next_batch(batch_size)

                    # draw random samples for latent layer
                    eps = np.random.normal(size=(batch_size, self.num_lvs))

                    # one step of optimization routine
                    sess.run(
                        self.train_step,
                        feed_dict={self.x: x[0], self.eps: eps})

                # print training updates
                if epochs_disp is not None and epoch % epochs_disp == 0:

                    # print updates using test set
                    x = data.test.next_batch(data.test.num_examples)
                    eps = np.random.normal(
                        size=(data.test.num_examples, self.num_lvs))
                    cost = sess.run(
                        self.cost,
                        feed_dict={self.x: x[0], self.eps: eps})
                    print('Epoch %03d:' % epoch)
                    print('   test cost = %2.5f' % cost)

                # save model checkpoints
                if epochs_ckpt is not None and epoch % epochs_ckpt == 0:
                    save_file = os.path.join(
                        output_dir, 'ckpts',
                        str('epoch_%05g.ckpt' % epoch))
                    self.save_model(sess, save_file)

                # save model summaries
                if epochs_summary is not None and \
                        epoch % epochs_summary == 0:
                    # output summaries using test set
                    x = data.test.next_batch(data.test.num_examples)
                    eps = np.random.normal(
                        size=(data.test.num_examples, self.num_lvs))
                    summary = sess.run(
                        self.merge_summaries,
                        feed_dict={self.x: x[0], self.eps: eps})
                    test_writer.add_summary(summary, epoch)

    def train_iters(
            self,
            sess,
            data=None,
            batch_size=128,
            iters_training=1000,
            iters_disp=None,
            iters_ckpt=None,
            iters_summary=None,
            output_dir=None):
        """
        Network training by specifying number of iterations rather than 
        epochs. Used for easily generating sample outputs during training

        Args:
            sess (tf.Session object): current session object to run graph
            data (DataReader object): input to network
            batch_size (int, optional): batch size used by the gradient
                descent-based optimizers
            iters_training (int, optional): number of iters for gradient 
                descent-based optimizers
            iters_disp (int, optional): number of iters between updates to 
                the console
            iters_ckpt (int, optional): number of iters between saving 
                checkpoint files
            iters_summary (int, optional): number of iters between saving
                network summary information 
            output_dir (string, optional): absolute path for saving checkpoint
                files and summary files; must be present if either iters_ckpt  
                or iters_summary is not 'None'.

        Returns:
            None

        Raises:
            InputError: If iters_ckpt is not None and output_dir is None
            InputError: If iters_summary is not None and output_dir is None

        """

        # check input
        if data is None:
            raise InputError('data reader must be specified')
        if iters_ckpt is not None and output_dir is None:
            raise InputError('output_dir must be specified to save model')
        if iters_summary is not None and output_dir is None:
            raise InputError('output_dir must be specified to save summaries')

        # initialize file writers
        if iters_summary is not None:
            test_writer = tf.summary.FileWriter(
                os.path.join(output_dir, 'summaries', 'test'),
                sess.graph)

        # begin training
        with self.graph.as_default():

            # start training loop
            for iter_ in range(iters_training):

                # get batch of data for this training step
                x = data.train.next_batch(batch_size)

                # draw random samples for latent layer
                eps = np.random.normal(size=(batch_size, self.num_lvs))

                # one step of optimization routine
                sess.run(
                    self.train_step,
                    feed_dict={self.x: x[0], self.eps: eps})

                # print training updates
                if iters_disp is not None and iter_ % iters_disp == 0:
                    # print updates using test set
                    x = data.test.next_batch(data.test.num_examples)
                    eps = np.random.normal(
                        size=(data.test.num_examples, self.num_lvs))
                    cost = sess.run(
                        self.cost,
                        feed_dict={self.x: x[0], self.eps: eps})
                    print('Iter %03d:' % iter_)
                    print('   test cost = %2.5f' % cost)

                # save model checkpoints
                if iters_ckpt is not None and iter_ % iters_ckpt == 0:
                    save_file = os.path.join(
                        output_dir, 'ckpts',
                        str('epoch_%05g.ckpt' % iter_))
                    self.save_model(sess, save_file)

                # save model summaries
                if iters_summary is not None and iter_ % iters_summary == 0:
                    # output summaries using test set
                    x = data.test.next_batch(data.test.num_examples)
                    eps = np.random.normal(
                        size=(data.test.num_examples, self.num_lvs))
                    summary = sess.run(
                        self.merge_summaries,
                        feed_dict={self.x: x[0], self.eps: eps})
                    test_writer.add_summary(summary, iter_)

    def generate(self, sess, z_mean=None):
        """ 
        Sample the network and generate an image 

        If z_mean is None, a random point is generated using the prior in
        the latent space, else z_mean is used as the point in latent space
        """

        if z_mean is None:
            z_mean = np.random.normal(size=self.num_lvs)

        return sess.run(self.x_recon, feed_dict={self.z: z_mean})

    def recognize(self, sess, x):
        """Transform a given input into its latent representation"""
        return sess.run(self.z_mean, feed_dict={self.x: x})

    def reconstruct(self, sess, x, eps):
        """Transform a given input into its reconstruction"""
        return sess.run(self.x_recon, feed_dict={self.x: x, self.eps: eps})

    def save_model(self, sess, save_file):
        """ 
        Save model parameters 

        Args:
            sess (tf.Session object): current session object to run graph
            save_file (str): full path to output file

        """

        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        self.saver.save(sess, save_file)
        print('Model saved to %s' % save_file)

    def load_model(self, sess, save_file):
        """ 
        Load previously saved model parameters 

        Args:
            sess (tf.Session object): current session object to run graph
            save_file (str): full path to saved model

        """

        if not os.path.isfile(save_file + '.meta'):
            raise InputError(str('%s is not a valid filename' % save_file))

        self.saver.restore(sess, save_file)
        print('Model loaded from %s' % save_file)


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super(InputError, self).__init__(message)
