#!/usr/bin/env python

"""Generative Adversarial Network class

@author: Matt Whiteway, March 2017
GAN class implements a generative adversarial network

"""

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append('..')
import utils.exceptions as exc


class GAN(object):
    """Generative Adversarial Network class

    Attributes:
        gen_input_size (int): size of input to generator network
        layers_gen (list of ints): size of each layer in generator, excluding
            input layer
        num_layers_gen (int): number of layers in generator
        disc_input_size (int): size of input to discriminator network 
        layers_disc (list of ints): size of each layer in discriminator, 
            excluding input layer
        num_layers_disc (int): number of layers in discriminator
        act_func (str): activation function for network layers

        weights_gen (list of tf.Variable): weights of the generator network
        biases_gen (list of tf.Variable): biases of the generator network
        weights_disc (list of tf.Variable): weights of the discriminator 
            network
        biases_disc (list of tf.Variable): biases of the discriminator network

        gen_input (tf placeholder): ph for latent variable input to generator
        img_gen (tf op): reconstructed input values
        img_real (tf placeholder): ph for input to model
        disc_gen (tf op): output of discriminator for generated images
        disc_real (tf op): output of discriminator for real images
        loss_gen (tf op): evaluates the cost function of the generative 
            network
        loss_disc (tf op): evaluates the cost function of the discriminative 
            network
        params_gen (tf op): returns the collection of variables in generative
            network 
        params_disc (tf op): returns the collection of variables in 
            discriminative network

        learning_rate (float): global learning rate used by gradient descent 
            optimizers
        train_step_gen (tf op): evaluates one training step for the generative
            network
        train_step_disc (tf op): evaluates one training step for the 
            discriminative network

        graph (tf.Graph): dataflow graph for the network
        saver (tf.train.Saver): for saving and restoring variables
        merge_summaries (tf op): op that merges all summary ops
        init (tf op): op that initializes global variables in graph 

    """

    def __init__(
            self,
            layers_gen=None,
            layers_disc=None,
            act_func='relu',
            learning_rate=1e-3):
        """ Constructor for GAN class 
        
        Args:
            layers_gen (list of ints): size of each layer in generator, 
                including output layer
            layers_disc (list of ints): size of each layer in discriminator, 
                including output layer
            act_func (str): activation function for network layers
                ['relu'] | 'sigmoid' | 'tanh' | 'linear' | 'softplus' | 'elu'
            learning_rate (scalar): global learning rate for gradient descent 
                methods

        Raises:
            InputError if layers_gen is not specified
            InputError if layers_disc is not specified
            InputError if act_func is not a valid string
            
        """

        # input checking
        if layers_gen is None:
            raise exc.InputError('Must specify layer sizes for generator')
        if layers_disc is None:
            raise exc.InputError('Must specify layer sizes for discrimantor')

        self.gen_input_size = layers_gen[0]
        self.layers_gen = layers_gen[1:]
        self.disc_input_size = layers_disc[0]
        self.layers_disc = layers_disc[1:]

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
            raise exc.InputError('Invalid activation function')

        self.learning_rate = learning_rate

        # define useful constants
        self.num_layers_gen = len(self.layers_gen)
        self.num_layers_disc = len(self.layers_disc)

        # for saving and restoring models
        self.graph = tf.Graph()  # must be initialized before graph creation

        with self.graph.as_default():

            # define pipeline for feeding data into model
            with tf.variable_scope('data'):
                self._initialize_data_pipeline()

            # initialize weights and create generator
            with tf.variable_scope('generator'):
                self._define_generator_network()

            # initialize weights and create discriminator
            self._define_discriminator()

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

    def _initialize_data_pipeline(self):
        """Create placeholders for input and random values"""

        self.gen_input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.gen_input_size],
            name='latent_vars_ph')
        self.img_real = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.disc_input_size],
            name='input_ph')

    def _define_generator_network(self):
        """Create a generator network to transform a random sample in the 
        latent space into an image
        """

        self.weights_gen = []
        self.biases_gen = []
        z_gen = [self.gen_input]
        for layer in range(self.num_layers_gen):
            with tf.variable_scope(str('layer_%01i' % layer)):

                # initialize weights
                if layer == 0:
                    in_size = self.gen_input_size
                else:
                    in_size = self.layers_gen[layer - 1]
                out_size = self.layers_gen[layer]
                self.weights_gen.append(tf.get_variable(
                    shape=[in_size, out_size],
                    name='weights',
                    initializer=tf.truncated_normal_initializer(stddev=0.1)))

                # initialize biases
                self.biases_gen.append(tf.get_variable(
                    initializer=tf.zeros(shape=[1, out_size]),
                    name='biases'))

                # calculate layer activations
                pre = tf.add(
                        tf.matmul(z_gen[layer], self.weights_gen[layer]),
                        self.biases_gen[layer])
                if layer == self.num_layers_gen - 1:
                    post = tf.nn.sigmoid(pre)
                else:
                    post = self.act_func(pre)
                z_gen.append(post)

                # save summaries of layer activations
                tf.summary.histogram('pre_act', pre)
                tf.summary.histogram('post_act', post)

        # define this for easier access later
        self.img_gen = z_gen[-1]

    def _define_discriminator(self):
        """Push real image and generated image thru discriminator network"""

        with tf.variable_scope('discriminator') as scope:
            self.disc_real = self._define_discriminator_network(self.img_real)
            scope.reuse_variables()
            self.disc_gen = self._define_discriminator_network(self.img_gen)

    def _define_discriminator_network(self, network_input):
        """Create a discriminator network to transform inputs into probability 
        that the input came from the training examples rather than the 
        generator network
        """
        # initialize weights and biases
        self.weights_disc = []
        self.biases_disc = []
        z_disc = [network_input]
        for layer in range(self.num_layers_disc):
            with tf.variable_scope(str('layer_%01i' % layer)):

                # initialize weights
                if layer == 0:
                    in_size = self.disc_input_size
                else:
                    in_size = self.layers_disc[layer - 1]
                out_size = self.layers_disc[layer]
                self.weights_disc.append(tf.get_variable(
                    shape=[in_size, out_size],
                    name='weights',
                    initializer=tf.truncated_normal_initializer(stddev=0.1)))
                # initialize biases
                self.biases_disc.append(tf.get_variable(
                    shape=[1, out_size],
                    name='biases',
                    initializer=tf.constant_initializer(0.0)))

                # calculate layer activations
                pre = tf.add(
                    tf.matmul(z_disc[layer], self.weights_disc[layer]),
                    self.biases_disc[layer])
                if layer == self.num_layers_disc - 1:
                    post = tf.identity(pre)  # use logits in cost function
                else:
                    post = self.act_func(pre)
                z_disc.append(post)

                # save summaries of layer activations
                tf.summary.histogram('pre_act', pre)
                tf.summary.histogram('post_act', post)

        return z_disc[-1]

    def _define_loss(self):
        """Define loss function that will be used to optimize model params"""

        # define generator loss
        with tf.variable_scope('generator'):
            self.loss_gen = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.disc_gen,
                    labels=tf.ones_like(self.disc_gen)))

        # define discriminator loss
        with tf.variable_scope('discriminator'):
            self.loss_disc = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.disc_real,
                    labels=tf.ones_like(self.disc_real)) +
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.disc_gen,
                    labels=tf.zeros_like(self.disc_gen)))

        # save summaries of losses
        tf.summary.scalar('loss_gen', self.loss_gen)
        tf.summary.scalar('loss_disc', self.loss_disc)

    def _define_optimizer(self):
        """
        Define one step of the optimization routine for both generator and
        discriminator networks
        """

        # set aside variable collections
        self.params_gen = self.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='generator')
        self.params_disc = self.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='discriminator')

        # define one step of the optimization routines
        self.train_step_gen = tf.train.AdamOptimizer(self.learning_rate). \
            minimize(self.loss_gen, var_list=self.params_gen)
        self.train_step_disc = tf.train.AdamOptimizer(self.learning_rate). \
            minimize(self.loss_disc, var_list=self.params_disc)

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
            InputError: If data is not specified
            InputError: If epochs_ckpt is not None and output_dir is None
            InputError: If epochs_summary is not None and output_dir is None

        """

        # check input
        if data is None:
            raise exc.InputError('data reader must be specified')
        if epochs_ckpt is not None and output_dir is None:
            raise exc.InputError('output_dir must be specified to save model')
        if epochs_summary is not None and output_dir is None:
            raise exc.InputError('output_dir must be specified to save ' +
                                 'summaries')

        # initialize file writers
        if epochs_summary is not None:
            test_writer = tf.summary.FileWriter(
                os.path.join(output_dir, 'summaries', 'test'),
                sess.graph)

        # define distribution of latent variables
        rand_dist = 'normal'

        with self.graph.as_default():

            # begin training
            for epoch in range(epochs_training):

                num_batches = int(data.train.num_examples / batch_size)

                # start training loop
                for batch in range(num_batches):

                    # one step of optimization routine for disc network
                    x = data.train.next_batch(batch_size)
                    if rand_dist == 'normal':
                        z = np.random.normal(
                            size=(batch_size, self.gen_input_size))
                    elif rand_dist == 'uniform':
                        z = np.random.uniform(
                            low=-1.0,
                            high=1.0,
                            size=(batch_size, self.gen_input_size))
                    sess.run(self.train_step_disc,
                             feed_dict={self.gen_input: z,
                                        self.img_real: x[0]})

                    # one step of optimization routine for gen network
                    if rand_dist == 'normal':
                        z = np.random.normal(
                            size=(batch_size, self.gen_input_size))
                    elif rand_dist == 'uniform':
                        z = np.random.uniform(
                            low=-1.0,
                            high=1.0,
                            size=(batch_size, self.gen_input_size))
                    sess.run(self.train_step_gen,
                             feed_dict={self.gen_input: z})

                # print training updates
                if epochs_disp is not None and epoch % epochs_disp == 0:
                    # print updates using test set
                    x = data.test.next_batch(data.test.num_examples)
                    if rand_dist == 'normal':
                        z = np.random.normal(
                            size=(data.test.num_examples, self.gen_input_size))
                    elif rand_dist == 'uniform':
                        z = np.random.uniform(
                            low=-1.0,
                            high=1.0,
                            size=(data.test.num_examples, self.gen_input_size))
                    [loss_gen, loss_disc] = sess.run(
                        [self.loss_gen, self.loss_disc],
                        feed_dict={self.gen_input: z,
                                   self.img_real: x[0]})
                    print('Epoch %03d:' % epoch)
                    print('   test loss gen = %2.5f' % loss_gen)
                    print('   test loss dis = %2.5f' % loss_disc)

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
                    if rand_dist == 'normal':
                        z = np.random.normal(
                            size=(data.test.num_examples, self.gen_input_size))
                    elif rand_dist == 'uniform':
                        z = np.random.uniform(
                            low=-1.0,
                            high=1.0,
                            size=(data.test.num_examples, self.gen_input_size))
                    summary = sess.run(
                        self.merge_summaries,
                        feed_dict={self.gen_input: z,
                                   self.img_real: x[0]})
                    test_writer.add_summary(summary, epoch)

    def generate(self, sess, z_mean=None, rand_dist='normal'):
        """
        Sample the network and generate an image 

        If z_mean is None, a random point is generated using the prior in the 
        latent space, else z_mean is used as the point in latent space
        """

        if z_mean is None:
            if rand_dist == 'normal':
                z_mean = np.random.normal(size=(1, self.gen_input_size))
            elif rand_dist == 'uniform':
                z_mean = np.random.uniform(
                    low=-1.0,
                    high=1.0,
                    size=(1, self.gen_input_size))

        return sess.run(self.img_gen, feed_dict={self.gen_input: z_mean})

    def save_model(self, sess, save_file=None):
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

    def load_model(self, sess, save_file=None):
        """Load previously saved model parameters 

        Args:
            sess (tf.Session object): current session object to run graph
            save_file (str): full path to saved model

        """

        if not os.path.isfile(save_file + '.meta'):
            raise exc.InputError(str('%s is not a valid filename' % save_file))

        self.saver.restore(sess, save_file)
        print('Model loaded from %s' % save_file)

# from https://github.com/AYLIEN/gan-intro/blob/master/gan.py
# import argparse
# def main(args):
#     model = GAN(
#         DataDistribution(),
#         GeneratorDistribution(range=8),
#         args.num_steps,
#         args.batch_size,
#         args.minibatch,
#         args.log_every,
#         args.anim
#     )
#     model.train()
#
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num-steps', type=int, default=1200,
#                         help='the number of training steps to take')
#     parser.add_argument('--batch-size', type=int, default=12,
#                         help='the batch size')
#     parser.add_argument('--minibatch', type=bool, default=False,
#                         help='use minibatch discrimination')
#     parser.add_argument('--log-every', type=int, default=10,
#                         help='print loss after this many steps')
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
# main(parse_args())