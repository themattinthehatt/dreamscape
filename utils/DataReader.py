#!/usr/bin/env python

"""
@author: Matt Whiteway, April 2017
DataReader class that handles mnist, cifar10 and imagenet datasets
"""


class DataReaderMNIST(object):
    """DataReader class for mnist"""

    def __init__(self, data_dir, one_hot=True):

        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets(data_dir, one_hot=one_hot)
        self.train = mnist.train
        self.test = mnist.test
        self.validation = mnist.validation


class DataReaderCIFAR(object):
    """DataReader class for cifar-10"""

    def __init__(self, data_dir, one_hot=True):

        raise NotImplementedError
