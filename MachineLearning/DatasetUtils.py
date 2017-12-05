# Note that this has some Otto dataset specific hardcodings.
# ref: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf
import os
import csv

try:
  import pandas as pd  # pylint: disable=g-import-not-at-top
except ImportError as e:
  print(e.msg)

feature_headers = [("feat_" + str(f), [0]) for f in range(1, 94)]
#feature_headers.insert(0, ('id', [0]))      # Format order as per the data source
feature_headers.append(('target', [""]))
defaults = collections.OrderedDict(feature_headers)
types = collections.OrderedDict((key, type(value[0])) for key, value in defaults.items())

# ref:
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/get_started/regression/imports85.py
def dataset(y_name, train_fraction=0.7):
  """Load the data as a (train,test) pair of `Dataset`.
  Each dataset generates (features_dict, label) pairs.
  Args:
    y_name: The name of the column to use as the label.
    train_fraction: A float, the fraction of data to use for training. The
        remainder will be used for evaluation.
  Returns:
    A (train,test) pair of `Datasets`
  """
  # Download and cache the data (hardcoded path for now)
  path = get_otto_data_filepath()

  # Define how the lines of the file should be parsed
  def decode_line(line):
    """Convert a csv line into a (features_dict,label) pair."""
    # Decode the line to a tuple of items based on the types of
    # csv_header.values().
    items = tf.decode_csv(line, list(defaults.values()))

    # Convert the keys and items to a dict.
    pairs = zip(defaults.keys(), items)
    features_dict = dict(pairs)

    # Remove the label from the features_dict
    label = features_dict.pop(y_name)

    return features_dict, label

  def in_training_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # If you randomly split the dataset you won't get the same split in both
    # sessions if you stop and restart training later.  Also a simple
    # random split won't work with a dataset that's too big to `.cache()` as
    # we are doing here.
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    # Use the hash bucket id as a random number that's deterministic per
    # example
    return bucket_id < int(train_fraction * num_buckets)

  def in_test_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # Items not in the training set are in the test set.
    # This line must use `~` instead of `not` beacuse `not` only works on
    # python
    # booleans but we are dealing with symbolic tensors.
    return ~in_training_set(line)

  base_dataset = tf.data.TextLineDataset(path)

  train = (base_dataset
           # Take only the training-set lines.
           .filter(in_training_set)
           # Cache data so you only read the file once.
           .cache()
           # Decode each line into a (features_dict, label) pair.
           .map(decode_line))

  # Do the same for the test-set.
  test = (base_dataset.filter(in_test_set).cache().map(decode_line))

  return train, test

def get_otto_data_filepath(filename = "train.csv"):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path + "\\Otto_data\\{}".format(filename) 


def raw_dataframe(path):
  """Load the data as a pd.DataFrame."""
  # Load it into a pandas dataframe
  df = pd.read_csv(path, header=0, names=types.keys(), dtype=types)

  return df

def load_data(y_name, train_fraction=0.7, seed=None):
  """Get the data set.
  Args:
    y_name: the column to return as the label.
    train_fraction: the fraction of the dataset to use for training.
    seed: The random seed to use when shuffling the data. `None` generates a
      unique shuffle every run.
  Returns:
    a pair of pairs where the first pair is the training data, and the second
    is the test data:
    `(x_train, y_train), (x_test, y_test) = get_dataset(...)`
    `x` contains a pandas DataFrame of features, while `y` contains the label
    array.
  """
  # Load the raw data columns.
  data = raw_dataframe('train.csv')

  # Delete rows with unknowns
  data = data.dropna()

  # Shuffle the data
  np.random.seed(seed)

  # Split the data into train/test subsets.
  x_train = data.sample(frac=train_fraction, random_state=seed)
  x_test = data.drop(x_train.index)

  # Extract the label from the features dataframe.
  y_train = x_train.pop(y_name).str.split('_').str[1]
  y_test =  x_test.pop(y_name).str.split('_').str[1]

  depth = 9

  # Convert label panda series to tensors
  t1 = tf.Variable(y_train.as_matrix(), dtype = tf.int32) 
  t2 = tf.Variable(y_test.as_matrix(), dtype = tf.int32) 

  # Split numbers from label strings
  y_train = tf.one_hot(t1, depth)
  y_test = tf.one_hot(t2, depth)

  return (x_train, y_train), (x_test, y_test)

d = load_data("target")
print(d)
