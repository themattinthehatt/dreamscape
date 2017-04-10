#!/usr/bin/env python

"""
@author: Matt Whiteway, March 2017
VAE class implements a variational autoencoder
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf

import DataReader as Data


class VAE(object):
    """Variational Autoencoder class"""

    def __init__(self,
                 layers_encoder=None, layer_latent=None, layers_decoder=None,
                 act_func=tf.nn.relu, batch_size=100, learning_rate=1e-3,
                 data_dir=None, data_type='mnist'):
        """Constructor for VAE class"""

        assert layers_encoder is not None, \
            'Must specify layer sizes for encoder'
        assert layer_latent is not None, \
            'Must specify number of latent dimensions'
        assert layers_decoder is not None, \
            'Must specify layer sizes for decoder'
        assert data_dir is not None, \
            'Must specify data directory'

        self.layers_encoder = layers_encoder
        self.layer_latent = layer_latent
        self.layers_decoder = layers_decoder
        self.layers_decoder.insert(0, layer_latent)  # helps with weight init

        self.act_func = act_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # define useful constants
        self.num_lvs = self.layer_latent
        self.num_layers_enc = len(self.layers_encoder)
        self.num_layers_dec = len(self.layers_decoder)

        # get data handler
        if data_type is 'mnist':
            self.data = Data.DataReaderMNIST(data_dir, one_hot=False)
        elif data_type is 'cifar':
            self.data = Data.DataReaderCIFAR(data_dir, one_hot=False)
        elif data_type is 'imagenet':
            self.data = Data.DataReaderImagenet(data_dir, one_hot=False)

        # for saving and restoring models
        self.graph = tf.Graph()  # must be initialized before graph creation

        with self.graph.as_default():
            # create placeholders for input and random values
            self.x = tf.placeholder(tf.float32,
                                    shape=[None, self.layers_encoder[0]])
            self.eps = tf.placeholder(tf.float32,
                                      shape=[None, self.num_lvs])

            # initialize weights and create model
            self._initialize_weights()
            self._define_recognition_network()
            self._define_generator_network()
            self._define_loss_optimizer()

            # add additional ops
            # for saving and restoring models
            self.saver = tf.train.Saver()  # must be init after var creation
            # add variable initialization op to graph
            # self.init = tf.global_variables_initializer()
            self.init = tf.initialize_all_variables()

    @staticmethod
    def weight_variable(shape, name='None'):
        """Utility method to clean up initialization"""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name='None'):
        """Utility method to clean up initialization"""
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial, name=name)

    def _initialize_weights(self):
        """Initialize weights and biases in model"""

        # initialize weights and biases in encoding model
        self.weights_enc = []
        self.biases_enc = []
        for i in range(self.num_layers_enc - 1):
            self.weights_enc.append(
                self.weight_variable([self.layers_encoder[i],
                                      self.layers_encoder[i + 1]],
                                     name=str('weights_enc_%02i' % i)))
            self.biases_enc.append(
                self.bias_variable([self.layers_encoder[i + 1]],
                                   name=str('biases_enc_%02i' % i)))

        # initialize weights and biases in decoding model
        self.weights_dec = []
        self.biases_dec = []
        for i in range(self.num_layers_dec - 1):
            self.weights_dec.append(
                self.weight_variable([self.layers_decoder[i],
                                      self.layers_decoder[i + 1]],
                                     name=str('weights_dec_%02i' % i)))
            self.biases_dec.append(
                self.bias_variable([self.layers_decoder[i + 1]],
                                   name=str('biases_dec_%02i' % i)))

        # initialize weights for means and stds of stochastic layer
        self.weights_mean = self.weight_variable(
            [self.layers_encoder[-1], self.num_lvs],
            name='weights_mean')
        self.biases_mean = self.bias_variable([self.num_lvs],
                                              name='biases_mean')
        self.weights_log_var = self.weight_variable(
            [self.layers_encoder[-1], self.num_lvs],
            name='weights_log_var')
        self.biases_log_var = self.bias_variable([self.num_lvs],
                                                 name='biases_log_var')

    def _define_recognition_network(self):
        """ 
        Create a recognition network to transform inputs into its latent 
        representation
        """

        # push data through the encoding function to determine mean and std
        # of latent vars
        z_enc = []
        for i in range(self.num_layers_enc):
            if i == 0:
                z_enc.append(self.x)
            else:
                z_enc.append(self.act_func(tf.add(
                    tf.matmul(z_enc[i - 1], self.weights_enc[i - 1]),
                    self.biases_enc[i - 1])))

        # weights to estimate mean of normally distributed latent vars
        self.z_mean = tf.add(tf.matmul(z_enc[-1], self.weights_mean),
                             self.biases_mean)
        # estimating log of the variance is easier since the latent loss has
        # a log determinant term
        self.z_log_var = tf.add(tf.matmul(z_enc[-1], self.weights_log_var),
                                self.biases_log_var)

    def _define_generator_network(self):
        """ 
        Create a generator network to transform a random sample
        in the latent space into an image
        """

        # transform estimated mean and log variance into a sampled value
        # of the latent state using z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_var)), self.eps))

        # push data through the decoding function to reconstruct data
        z_dec = []
        for i in range(self.num_layers_dec - 1):
            if i == 0:
                z_dec.append(self.act_func(tf.add(
                    tf.matmul(self.z, self.weights_dec[i]),
                    self.biases_dec[i])))
            else:
                z_dec.append(self.act_func(tf.add(
                    tf.matmul(z_dec[i - 1], self.weights_dec[i]),
                    self.biases_dec[i])))

        # define this for easier access later
        self.x_recon = z_dec[-1]

    def _define_loss_optimizer(self):
        """ 
        Create the loss function that will be used to optimize
        model parameters as well as define the optimizer
        """

        # define reconstruction loss
        loss_recon = 0.5 * tf.reduce_sum(tf.square(self.x_recon - self.x), 1)

        # define latent loss
        loss_latent = 0.5 * tf.reduce_sum(tf.exp(self.z_log_var)
                                          + tf.square(self.z_mean)
                                          - 1 - self.z_log_var, 1)

        # define cost
        self.cost = tf.reduce_mean(loss_recon + loss_latent)

        # define one step of the optimization routine
        self.train_step = tf.train.AdamOptimizer(self.learning_rate). \
            minimize(self.cost)

    def train(self, sess, batch_size=None,
              training_epochs=75,
              display_epochs=1):
        """Network training by specifying epochs"""

        with self.graph.as_default():

            batch_size = self.batch_size if batch_size is None else batch_size

            for epoch in range(training_epochs):

                num_batches = int(self.data.train.num_examples / batch_size)

                for batch in range(num_batches):
                    # get batch of data for this training step
                    x = self.data.train.next_batch(batch_size)

                    # draw random samples for latent layer
                    eps = np.random.normal(
                        size=(self.batch_size, self.num_lvs))

                    # one step of optimization routine
                    sess.run(self.train_step, feed_dict={self.x: x[0],
                                                         self.eps: eps})

                # print training updates
                if display_epochs is not None and epoch % display_epochs == 0:
                    train_accuracy = sess.run(self.cost,
                                              feed_dict={self.x: x[0],
                                                         self.eps: eps})
                    print('Epoch %03d: cost = %2.5f' % (epoch, train_accuracy))

    def train_iters(self, sess, batch_size=None,
                    training_iters=20000,
                    display_iters=2000):
        """
        Network training by specifying number of iterations rather than epochs
        Used for easily generating sample outputs during training
        """

        batch_size = self.batch_size if batch_size is None else batch_size

        for tr_iter in range(training_iters):
            # get batch of data for this training step
            x = self.data.train.next_batch(batch_size)

            # draw random samples for latent layer
            eps = np.random.normal(size=(self.batch_size, self.num_lvs))

            # one step of optimization routine
            sess.run(self.train_step, feed_dict={self.x: x[0],
                                                 self.eps: eps})

        # print training updates
        if display_iters is not None and tr_iter % display_iters == 0:
            train_accuracy = sess.run(self.cost, feed_dict={self.x: x[0],
                                                            self.eps: eps})
            print('Iter %03d: cost = %2.5f' % (tr_iter, train_accuracy))

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

    def save_model(self, sess, save_file=None):
        """Save model parameters"""

        assert save_file, 'Must specify filename to save model'
        self.saver.save(sess, save_file)
        print('Model saved to %s' % save_file)

    def load_model(self, sess, save_file=None):
        """Load previously saved model parameters"""

        assert save_file, 'Must specify model location'
        self.saver.restore(sess, save_file)
        print('Model loaded from %s' % save_file)
