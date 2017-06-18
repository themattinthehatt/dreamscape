"""Functions for downloading and reading CIFAR10 data
This file is mostly copied from tensorflow's mnist input file, found on github
under:
tensorflow/examples/tutorials/mnist/input_data.py (r0.7)

The extracted data is held in DataSet objects as numpy arrays; for a fancier
(read: scalable) alternative using queues, see 
tensorflow/models/image/cifar10_input.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import sys
import os

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
TRAIN_SIZE = 45000
TEST_SIZE = 10000
VALIDATION_SIZE = 5000
NUM_FILES_TRAIN = 5
NUM_FILES_TEST = 1
IMAGES_PER_FILE = 10000

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10
IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH
LABEL_BYTES = 1


def maybe_download_and_extract(work_directory):
    """Download the tarball from Alex's website."""

    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filename = SOURCE_URL.split('/')[-1]
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' %
                (filename, float(count * block_size) /
                 float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(SOURCE_URL,
                                                 filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(work_directory, 'cifar-10-batches-bin/')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(work_directory)

    return extracted_dir_path


def extract_images_and_labels(filepath, one_hot=False, training=True):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""

    if training:

        total_records = TRAIN_SIZE + VALIDATION_SIZE
        images = np.zeros(
            shape=(total_records, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH),
            dtype='uint8')
        labels = np.zeros(shape=(total_records, LABEL_BYTES), dtype='int32')

        # loop through data files
        count = 0
        for i in range(NUM_FILES_TRAIN):
            # Load the images and class-numbers from the data-file.
            filename = filepath + str('data_batch_%i.bin' % (i + 1))
            print('Extracting', filename)
            with open(filename, 'rb') as bytestream:
                for index in range(IMAGES_PER_FILE):
                    # read single label and image from bytestream
                    buf = bytestream.read(LABEL_BYTES + IMAGE_BYTES)
                    data = np.frombuffer(buf, dtype=np.uint8)
                    # first byte represents the label; convert unit8->int32
                    labels[count] = data[0].astype('int32')
                    # the remaining bytes represent the image, which is stored
                    # as depth * height * width
                    image_reshape = np.reshape(
                        data[1:],
                        [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
                    # convert from [depth, height, width] to
                    # [height, width, depth]
                    images[count, :, :, :] = np.transpose(
                        image_reshape, [1, 2, 0])
                    count += 1

    else:

        total_records = TEST_SIZE
        images = np.zeros(
            shape=(total_records, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH),
            dtype='uint8')
        labels = np.zeros(shape=(total_records, LABEL_BYTES), dtype='int32')

        count = 0
        # Load the images and class-numbers from the data-file.
        filename = filepath + 'test_batch.bin'
        print('Extracting', filename)
        with open(filename, 'rb') as bytestream:
            for index in range(IMAGES_PER_FILE):
                # read single label and image from bytestream
                buf = bytestream.read(LABEL_BYTES + IMAGE_BYTES)
                data = np.frombuffer(buf, dtype=np.uint8)
                # first byte represents the label; convert unit8->int32
                labels[count] = data[0].astype('int32')
                # the remaining bytes represent the image, which is stored
                # as depth * height * width
                image_reshape = np.reshape(
                    data[1:],
                    [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
                # convert from [depth, height, width] to
                # [height, width, depth]
                images[index, :, :, :] = np.transpose(
                    image_reshape, [1, 2, 0])
                count += 1

    if one_hot:
        labels = dense_to_one_hot(labels, NUM_CLASSES)

    return images, labels


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):

    def __init__(self, images, labels, fake_data=False, one_hot=False,
                 dtype=tf.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """

        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError(
                'Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape,
                                                       labels.shape))
        self._num_examples = images.shape[0]
        self._width = images.shape[1]
        self._height = images.shape[2]
        self._depth = images.shape[3]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns*depth]
        assert images.shape[3] == IMAGE_DEPTH
        images = images.reshape(
            images.shape[0],
            images.shape[1] * images.shape[2] * images.shape[3])
        if dtype == tf.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def depth(self):
        return self._depth

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * self._width * self._height * self._depth
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=tf.float32):
    class DataSets(object):
        pass
    data_sets = DataSets()

    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot,
                           dtype=dtype)
        data_sets.train = fake()
        data_sets.validation = fake()
        data_sets.test = fake()
        return data_sets

    local_file = maybe_download_and_extract(train_dir)

    train_images, train_labels = extract_images_and_labels(
        local_file, one_hot=one_hot, training=True)

    test_images, test_labels = extract_images_and_labels(
        local_file, one_hot=one_hot, training=False)

    val_images = train_images[:VALIDATION_SIZE]
    val_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
    data_sets.validation = DataSet(val_images, val_labels, dtype=dtype)
    data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

    return data_sets
