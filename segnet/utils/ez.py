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

  import os
  import matplotlib.pyplot as plt
  plt.style.use("ggplot")
  import tensorflow as tf
  from typing import Tuple, Optional, Union, Dict, Any, NoReturn

  @staticmethod
  def assert_isdir(path: str) -> str:
    """
      Returns argument `path` if it is a directory, 
      raises an exception otherwise.   
    """
    if os.path.isdir(path):
      return path
    else:
      raise Exception(f"Path '{path}' is not a directory.")
  
  @staticmethod
  def ls(path: str) -> List[str]:
    """ Wrapper for os.listdir(path) """
    return os.listdir(path)

  @staticmethod
  def assert_isfile(path: str) -> str:
    """ Docstring """
    if os.path.isfile(path):
      return path
    else:
      raise Exception(f"File '{path}' is not a regular file.")    

  def __init__(
      self, 
      model: tf.keras.Model,
      name: str,
      snapshot_dir: str,
      history_dir: str,
      data_path: str,
      log_file_path: str, 
      author: Optional[str] = None,
      seed: int = 1
  ) -> None:
    self._model: tf.keras.Model = model
    self.__datetime: datetime.datetime = datetime.datetime.utcnow()
    self._date: str = str(self.__datetime).split(".")[0]
    self._name: str = name
    self._snapshot_dir: str = self.assert_isdir(snapshot_dir)
    self._history_dir: str = self.assert_isdir(history_dir)
    self._data_path: str = self.assert_isdir(data_path)
    self._log_file_path: str = self.assert_isfile(log_file_path)
    self._img_path: str = os.path.join(self._data_path, 'imgs')
    self._msks_path: str = os.path.join(self._data_path, 'msks')
    self.__seed: int = 1
    # Optional args :
    if author is not None:
      self._author: str = author
    else:
      self._author: str = "Unknown"
    # Decorate __compile method
    self.__compile = timing.time_log(self._log_file_path)(self.__compile)
    

  def __getitem__(self, key):
    try:
      return getattr(self, f"_{key}")
    except AttributeError:
      print(f"Attribute `{key}` is not yet defined for this object (or does not exist)")
      print(f"Defined attributes are: {self.keys}")

  @property
  def keys(self) -> List[Any]:
    """ Get the keys (as a list) that can be used to index the Segmed object """
    return sorted(list(map(lambda x: x[1:], self.__dict__.keys())))

  @property
  def items(self) -> Dict[str,Any]:
    """ Get the items `{ key: value }` pairs oh the class. """
    return { key: self[key] for key in self.keys }

  @property
  def values(self) -> List[Any]:
    """ Get the values, i.e. object properties. """ 
    return [ self[key] for key in self.keys ]

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
  def images_path(self) -> str:
    """ Path where images are stored. 
    Following the Keras convention it should contain a
    directory named `images`.
    """
    return self._img_path

  @property
  def masks_path(self) -> str:
    """ Path where segmentation masks are stored. 
    Following the Keras convention it should contain a
    directory named `masks`.
    """
    return self._msks_path

  @property
  def images(self) -> List[str]:
    """ A list of all the images contained in `self.images_path`/images/ """
    return self.ls(os.path.join(self.images_path, 'images'))

  @property
  def masks(self) -> List[str]:
    """ A list of all the segmentation masks contained in `self.masks_path`/masks/ """
    return self.ls(os.path.join(self.masks_path, 'masks'))
  
  @property
  def snapshots(self) -> List[str]:
    """ List of files (snapshots) found in `self._snapshot_dir` """
    return self.ls(self._snapshot_dir)

  @property 
  def historiae(self) -> List[str]:
    """ List of files (training histories) found in `self._history_dir` """
    return self.ls(self._history_dir)

  def create_train_test_generators(
      self, 
      data_gen_args: Optional[Dict[str, Any]] = None,
      hyper_params: Optional[Dict[str, Any]] =  None
  ) -> NoReturn:
    """ Generate the train/test split. 
      data_gen_args defaults to the following Dict :
        {
          "rescale": 1.0 / 255.0,
          "validation_split": 0.2,
          "dtype": tf.float32,
        }
    """
    # Rescale and convert to float32 both subsets
    if data_gen_args is None:
      self._data_gen_args = {
          "rescale": 1.0 / 255.0,
          "validation_split": 0.2,
          "dtype": tf.float32,
      }
    else:
      self._data_gen_args = data_gen_args

    if hyper_params is None:
      self._hyper_params = dict(
        batch_size = 8,
        epochs = 80,
        steps_per_epoch=10
      )

    # Crea the training generators with the defined transformations
    self.__image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**self._data_gen_args)
    self.__mask_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(**self._data_gen_args)
    
    # Take images from directories
    self._image_generator_train = self.__image_datagen.flow_from_directory(
        self._img_path,
        class_mode=None,
        color_mode="rgb",
        batch_size=self._hyper_params["batch_size"],
        seed=self.__seed,
        subset="training",
    )
    self._mask_generator_train = self.__mask_datagen.flow_from_directory(
        self._msks_path,
        class_mode=None,
        color_mode="grayscale",
        batch_size=self._hyper_params["batch_size"],
        seed=self.__seed,
        subset="training",
    )
    # Combine both generators
    # This needs to be a generator of generators, see the following
    # https://github.com/tensorflow/tensorflow/issues/32357
    self._train_generator = (
        pair for pair in zip(self._image_generator_train, self._mask_generator_train)
    )

    # And now the validation generators
    self._image_generator_val = self.__image_datagen.flow_from_directory(
        self._img_path,
        class_mode=None,
        color_mode="rgb",
        batch_size=self._hyper_params["batch_size"],
        seed=self.__seed,
        subset="validation",
    )

    self._mask_generator_val = self.__mask_datagen.flow_from_directory(
        self._msks_path,
        class_mode=None,
        color_mode="grayscale",
        batch_size=self._hyper_params["batch_size"],
        seed=self.__seed,
        subset="validation",
    )

    # Combine both generators, with same issue as before
    self._val_generator = (
        pair for pair in zip(self._image_generator_val, self._mask_generator_val)
    )

  def show_img_and_mask(self, n: int = 5) -> NoReturn:
    """ Show `n` images and segmentation maps, side by side
    to verify that they are batched together.
    """
    for i, pair in zip(range(n), self._train_generator):
      plt.figure(0)
      plt.subplot(1, 2, 1)
      plt.imshow(pair[0][0, ...])
      plt.subplot(1, 2, 2)
      plt.imshow(pair[1][0, ..., 0])
      plt.show()

    # All code from here needs real integration : 
    
  def create_custom_callback(
      self, 
      model_checkpoint_kw: Optional[Dict[str,Any]] = None
  ) -> NoReturn:
    """ Define the checkpoint callback.
      Parameters (keyword arguments) used to call 
      `tf.keras.callbacks.ModelCheckpoint()` can be 
      specified as a dictionary (parameter `model_checkpoint_kw`).

      Parameters:
        model_checkpoint_kw : Optional, defaults to:
          Dict(
            self._model_file, 
            monitor=monitor, 
            verbose=1, 
            save_best_only=True, 
            mode="max"
          )
    """
    # Define the checkpoint callback, always maximum mode for custom metrics
    if model_checkpoint_kw is None:
      model_checkpoint_kw = dict(
        self._model_file, 
        monitor=monitor, 
        verbose=1, 
        save_best_only=True, 
        mode="max"
      )
      
    self._checkpoint = tf.keras.callbacks.ModelCheckpoint(**model_checkpoint_kw)
    
  def compile(
      self,
      compiling_kw: Optional[Dict[str,Any]] = None
  ) -> NoReturn:
    """ This eventually will be a wrapper """ 
    if compiling_kw is None:
      compiling_kw = dict(
          loss=tf.keras.losses.binary_crossentropy,
          optimizer=tf.keras.optimizers.Adam(),
          metrics=[
            mts.jaccard_index, mts.dice_coef, 
            mts.O_Rate, mts.U_Rate, mts.Err_rate
          ]
      )
    self.__compile(**compiling_kw)

  def __compile(self, **compiling_kw):
    # Compile the model with custom params
    self._model.compile(**compiling_kw)

  def train(
      self, 
      compiling_kw: Dict[str,Any] = None,
      
  ) -> NoReturn:
    """ Docstring """
    self.compile()
    self.create_train_test_generators()
    
    # Create history 
    self._history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[checkpoint],
        verbose=1,
        validation_data=val_generator,
        validation_steps=steps_per_epoch,
    )

    

