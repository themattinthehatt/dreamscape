#!/usr/bin/env python

"""
@author: Matt Whiteway, March 2017
GAN class implements a generative adversarial network
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf


class GAN(object):
    """Generative Adversarial Network class"""

    def __init__(self,
                 layers_gen=None, layers_disc=None,
                 act_func=tf.nn.relu, batch_size=100, learning_rate=1e-3):
        """ Constructor for GAN class """

        assert layers_gen is not None, \
            'Must specify layer sizes for generator'
        assert layers_disc is not None, \
            'Must specify layer sizes for discriminator'

        self.layers_gen = layers_gen
        self.layers_disc = layers_disc

        self.act_func = act_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # define useful constants
        self.num_layers_gen = len(self.layers_gen)
        self.num_layers_disc = len(self.layers_disc)

        # for saving and restoring models
        self.graph = tf.Graph()  # must be initialized before graph creation

        with self.graph.as_default():
            # initialize weights/placeholders and create model
            self._initialize_weights()
            self._define_generator_network()
            self._define_discriminator()
            self._define_loss_optimizer()

            # add additional ops
            # for saving and restoring models
            self.saver = tf.train.Saver()  # must be init after var creation
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()

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

        # initialize weights and biases in generator network
        with tf.variable_scope('gen'):
            self.gen_input = tf.placeholder(tf.float32,
                                            shape=[None, self.layers_gen[0]],
                                            name='Z')
            self.weights_gen = []
            self.biases_gen = []
            for i in range(self.num_layers_gen - 1):
                self.weights_gen.append(
                    self.weight_variable([self.layers_gen[i],
                                          self.layers_gen[i + 1]],
                                         name=str('weights_%02i' % i)))
                self.biases_gen.append(
                    self.bias_variable([self.layers_gen[i + 1]],
                                       name=str('biases_%02i' % i)))

        # initialize weights and biases in discriminator network
        with tf.variable_scope('disc'):
            self.img_real = tf.placeholder(tf.float32,
                                           shape=[None, self.layers_disc[0]],
                                           name='X')
            self.weights_disc = []
            self.biases_disc = []
            for i in range(self.num_layers_disc - 1):
                self.weights_disc.append(
                    self.weight_variable([self.layers_disc[i],
                                          self.layers_disc[i + 1]],
                                         name=str('weights_%02i' % i)))
                self.biases_disc.append(
                    self.bias_variable([self.layers_disc[i + 1]],
                                       name=str('biases_%02i' % i)))

    def _define_generator_network(self):
        """ 
        Create a generator network to transform a random sample
        in the latent space into an image
        """

        # push data through the generator network to reconstruct data
        with tf.variable_scope('gen'):
            z_gen = []
            for i in range(self.num_layers_gen - 1):
                if i == 0:
                    z_gen.append(self.act_func(tf.add(
                        tf.matmul(self.gen_input, self.weights_gen[i]),
                        self.biases_gen[i])))
                else:
                    z_gen.append(self.act_func(tf.add(
                        tf.matmul(z_gen[i - 1], self.weights_gen[i]),
                        self.biases_gen[i])))

            # define this for easier access later
            self.img_gen = z_gen[-1]

    def _define_discriminator(self):
        """ 
        Push real image and generated image through discriminator network
        """

        with tf.variable_scope('disc') as scope:
            self.disc_real = self._define_discriminator_network(self.img_real)
            scope.reuse_variables()
            self.disc_gen = self._define_discriminator_network(self.img_gen)

    def _define_discriminator_network(self, network_input):
        """ 
        Create a discriminator network to transform inputs into probability 
        that the input came from the training examples rather than the 
        generator network
        """

        z_disc = []
        for i in range(self.num_layers_disc - 1):
            if i == 0:
                z_disc.append(self.act_func(tf.add(
                    tf.matmul(network_input, self.weights_disc[i]),
                    self.biases_disc[i])))
            elif i == self.num_layers_disc - 2:
                z_disc.append(tf.nn.sigmoid(tf.add(
                    tf.matmul(z_disc[i - 1], self.weights_disc[i]),
                    self.biases_disc[i])))
            else:
                z_disc.append(self.act_func(tf.add(
                    tf.matmul(z_disc[i - 1], self.weights_disc[i]),
                    self.biases_disc[i])))

        return z_disc[-1]

    def _define_loss_optimizer(self):
        """ 
        Create the loss function that will be used to optimize model parameters
        as well as define the optimizers
        """

        # define generator loss
        self.loss_gen = -0.5 * tf.reduce_mean(tf.log(self.disc_gen))

        # define discriminator loss
        self.loss_disc = -0.5 * tf.reduce_mean(tf.log(self.disc_real) +
                                               tf.log(1 - self.disc_gen))

        # set aside variable collections
        self.params_gen = self.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='gen')
        self.params_disc = self.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='disc')

        # define one step of the optimization routines
        self.train_step_gen = tf.train.AdamOptimizer(self.learning_rate). \
            minimize(self.loss_gen, var_list=self.params_gen)
        self.train_step_disc = tf.train.AdamOptimizer(self.learning_rate). \
            minimize(self.loss_disc, var_list=self.params_disc)

    def train(self, sess, data=None,
              batch_size=None,
              training_epochs=75,
              display_epochs=1):
        """Network training by specifying epochs"""

        assert data is not None, 'Must specify data reader object'

        with self.graph.as_default():

            batch_size = self.batch_size if batch_size is None else batch_size

            for epoch in range(training_epochs):

                num_batches = int(data.train.num_examples / batch_size)

                for batch in range(num_batches):

                    # one step of optimization routine for disc network
                    x = data.train.next_batch(batch_size)
                    z = np.random.randn(batch_size, self.layers_gen[0])
                    sess.run(self.train_step_disc,
                             feed_dict={self.gen_input: z,
                                        self.img_real: x[0]})

                    # one step of optimization routine for gen network
                    z = np.random.randn(batch_size, self.layers_gen[0])
                    sess.run(self.train_step_gen,
                             feed_dict={self.gen_input: z})

                # print training updates
                if display_epochs is not None and epoch % display_epochs == 0:
                    train_accuracy_disc = sess.run(
                        self.train_step_disc,
                        feed_dict={self.gen_input: z,
                                   self.img_real: x[0]})
                    train_accuracy_gen = sess.run(
                        self.train_step_gen,
                        feed_dict={self.gen_input: z})

                    print('Epoch %03d: disc cost = %2.5f' %
                          (epoch, train_accuracy_disc))
                    print('Epoch %03d: gen cost = %2.5f' %
                          (epoch, train_accuracy_gen))
                    print('')

    def train_iters(self, sess, data,
                    batch_size=None,
                    training_iters=20000,
                    display_iters=2000):
        """ Network training by specifying number of iterations rather than 
        epochs
        Used for easily generating sample outputs during training
        """

        assert data is not None, 'Must specify data reader object'

        with self.graph.as_default():

            batch_size = self.batch_size if batch_size is None else batch_size

            for tr_iter in range(training_iters):
                # get batch of data for this training step
                x = data.train.next_batch(batch_size)

                # draw random samples for latent layer
                eps = np.random.normal(size=(batch_size, self.layers_gen[0]))

                # one step of optimization routine
                sess.run(self.train_step, feed_dict={self.x: x[0],
                                                     self.eps: eps})

            # print training updates
            if display_iters is not None and tr_iter % display_iters == 0:
                train_accuracy = sess.run(self.cost, feed_dict={self.x: x[0],
                                                                self.eps: eps})
                print("Iter %03d: cost = %2.5f" % (tr_iter, train_accuracy))

    def save_model(self, sess, save_file=None):
        """ Save model parameters """

        assert save_file, 'Must specify filename to save model'
        self.saver.save(sess, save_file)
        print('Model saved to %s' % save_file)

    def load_model(self, sess, save_file=None):
        """ Load previously saved model parameters """

        assert save_file, 'Must specify model location'
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