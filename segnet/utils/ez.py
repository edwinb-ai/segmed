"""
    Easiest way to work with a Segmed Model.
"""

### System-related
import sys
import os
import datetime
from typing import Optional, List, Tuple, Any, NoReturn, Callable, Union, Type
#import importlib.util
###############################################################

### Machine learning specific
#import segmentation_models as sm
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import keras.models
import tensorflow as tf
###############################################################

### In-Out
from skimage import io as skio
import json
import pandas as pd
import h5py
import glob
###############################################################

### Visualization
import seaborn as sns
###############################################################

### Numerical
import numpy as np
###############################################################

### Repo-specific (segnet)
from segnet.train import train_segnet
from segnet.models import unet
from segnet.models import multiresunet as mru
from segnet.models import multiresunet2 as mru2
from segnet.models import multiresunet3 as mru3
from segnet.utils import timing
from segnet.metrics import metrics as mts


class Segmed(object):
  """
    Deep Learning Model Generator Object.
  """

  @staticmethod
  def assert_isdir(path: str):
    """
      Returns argument `path` if it is a directory,
      raises an exception otherwise.
    """
    if os.path.isdir(path):
      return path
    else:
      raise Exception(f"Path '{path}' is not a directory.")

  @staticmethod
  def ls(path: str):
    """ Wrapper for os.listdir(path) """
    return os.listdir(path)

  def __init__(
      self,
      model: tf.keras.Model,
      name: str,
      snapshot_dir: str,
      history_dir: str,
      data_path: str,
      author: Optional[str] = None
  ) -> None:
    self._model = model
    self.__datetime: datetime.datetime = datetime.datetime.utcnow()
    self._date: str = str(self.__datetime).split(".")[0]
    self._name: str = name
    self._snapshot_dir: str = self.assert_isdir(snapshot_dir)
    self._history_dir: str = self.assert_isdir(history_dir)
    self._data_path: str = self.assert_isdir(data_path)
    self._img_path: str = os.path.join(self._data_path, 'imgs')
    self._msks_path: str = os.path.join(self._data_path, 'msks')
    if author is not None:
      self._author: str = author
    else:
      self._author: str = "Unknown"

  def __getitem__(self, key):
    return getattr(self, f"_{key}")

  @property
  def name(self):
    """ Name of the model, composed of: name-(author)-(date) """
    return f"{self._name}-({self._author})-({self._date})"

  @property
  def snapshot_file(self):
    """ Name (full path) of the model snapshot file (.h5 format) """
    return os.path.join(self._snapshot_dir, f"{self.name}.h5")

  @property
  def history_file(self):
    """ Name (full path) of the training history file (.csv format) """
    return os.path.join(self._history_dir, f"{self.name}.csv")

  @property
  def images_path(self):
    """ Path where images are stored.
    Following the Keras convention it should contain a
    directory named `images`.
    """
    return self._img_path

  @property
  def masks_path(self):
    """ Path where segmentation masks are stored.
    Following the Keras convention it should contain a
    directory named `masks`.
    """
    return self._msks_path

  @property
  def images(self):
    """ A list of all the images contained in `self.images_path`/images/ """
    return self.ls(os.path.join(self.images_path, 'images'))

  @property
  def masks(self):
    """ A list of all the segmentation masks contained in `self.masks_path`/masks/ """
    return self.ls(os.path.join(self.masks_path, 'masks'))


