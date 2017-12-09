# Note that this has some Otto dataset specific hardcodings.
# ref: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
import collections
import numpy as np
import tensorflow as tf
import os       # For working with filesystm
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

def get_otto_data_filepath(filename = "train.csv"):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path + "\\Otto_data\\{}".format(filename) 


def raw_dataframe(path):
  """Load the data as a pd.DataFrame."""
  # Load it into a pandas dataframe
  df = pd.read_csv(path, header=0, names=defaults.keys(), dtype=types)

  return df

def load_data(y_name, filename):
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
  data = raw_dataframe(filename)

  # Delete rows with unknowns
  x_train = data.dropna()

  # Extract the label from the features dataframe.
  y_train = x_train.pop(y_name).str.split('_').str[1]

  depth = 10    # For one-hot vector

  # Convert label panda series to tensors
  t1 = tf.Variable(y_train.as_matrix(), dtype = tf.int32) 

  # Split numbers from label strings
  y_train = tf.one_hot(t1, depth)

  return (x_train, y_train)

# Tester
d = load_data("target", 'test.csv')
#print(d)
