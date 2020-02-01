
# System and utility :
import platform,socket,re,uuid,json,psutil # for getSystemInfo method
import os
import GPUtil
import datetime
from typing import Tuple, Optional, Union, Dict, Any, NoReturn, Callable, List

# Data-handling/Plotting :
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

# Repo-specific imports :
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

  # Definition of static methods, utility functions logically
  # related to the class, but not semantically.
  @staticmethod
  def assert_isdir(path: str) -> str:
    """ Returns argument `path` if it is a directory, 
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
    """ Returns argument `path` if it is a regular file, 
    raises an exception otherwise.   
    """
    if os.path.isfile(path):
      return path
    else:
      raise Exception(f"File '{path}' is not a regular file.")    

  @staticmethod
  def json_cast(x: Any) -> Union[str,Any]:
    """ Verify via (duck-typing) if an object is JSON-serialisable
    
    Returns :
          x,  if parameter `x` is JSON-serialisable, 
    
      str(x), if json.dumps(x) throws an exception.
    """
    if timing.is_jsonable(x):
      return x
    else:
      return str(x)

  def _getSystemInfo() -> Dict[str,Union[str,int]]:
    """ Get SystemInfo as a dictionary.
    Original code snippet can be found at :
    https://stackoverflow.com/questions/3103178/how-to-get-the-system-info-with-python 
    """
    try:
      info={}
      info['platform']=platform.system()
      info['platform-release']=platform.release()
      info['platform-version']=platform.version()
      info['architecture']=platform.machine()
      info['hostname']=socket.gethostname()
      info['processor']=platform.processor()
      info['ram (GB)']=round(psutil.virtual_memory().total / (1024.0 **3))
      return info
    except Exception as e:
      logging.exception(e)

  def _get_gpu() -> List[Dict[str,Union[str,float,int]]]:
    """ Get a List of dictionaries, describing each one of the GPUs found, if any. """
    gpus: List[GPUtil.GPUtil.GPU] = GPUtil.getGPUs()
    gpu_list: List[Dict[str,Union[str,float,int]]] = []

    if len(gpus) == 0:
      return [{}]

    for gpu in gpus:
      gpu_list.append({
          "ID": gpu.id,
          "Name": gpu.name,
          "Serial": gpu.serial,
          "UUID": gpu.uuid,
          "GPU temp.": gpu.load*100,
          "GPU util %": gpu.memoryUtil*100,
          "Memory Total (MB)": gpu.memoryTotal,
          "Memory used (MB)": gpu.memoryUsed,
          "Memory Free (MB)": gpu.memoryFree,
          "Display Mode": gpu.display_mode,
          "Display Active": gpu.display_active
      })
    return gpu_list
  # End of static methods definition.

  # Class attributes which will be used if necessary parameters are 
  # not passed to different methods.
  # Decided to use dict literals instead of the constructor because it's faster.
  __data_gen_args: Dict[str,Any] = {
      "rescale": 1.0 / 255.0,
      "validation_split": 0.2,
      "dtype": tf.float32,
  }

  __hyper_params: Dict[str,Any] = {
      "batch_size": 8,
      "epochs": 10,
      "steps_per_epoch": 10
  }

  __model_checkpoint_kw: Dict[str,Any] = {
      "monitor": "val_jaccard_index", 
      "verbose": 1, 
      "save_best_only": True, 
      "mode": "max"
  }

  __compiling_kw: Dict[str,Any] = {
      "loss": tf.keras.losses.binary_crossentropy,
      "optimizer": tf.keras.optimizers.Adam(),
      "metrics": [
        mts.jaccard_index, mts.dice_coef, 
        mts.O_Rate, mts.U_Rate, mts.Err_rate
      ]
  }
  # End of class attributes.

  def __init__(
      self, 
      model: tf.keras.Model,
      name: str,
      base_dir: str,
      data_path: str,
      author: str,
      seed: int = 1,
      plt_style: Optional[str] = None
  ) -> None:
    # Set style for plots :
    plt_style = plt_style or "ggplot"
    plt.style.use(plt_style)
    # Set instance attributes :
    ## Identifiers :
    self._model: tf.keras.Model = model
    self.__datetime: datetime.datetime = datetime.datetime.utcnow()
    self._date: str = str(self.__datetime).split(".")[0]
    self._name: str = name
    self._author: str = author
    ## Directories :
    ### Model :
    self._base_dir: str = self.assert_isdir(base_dir)
    self._instance_dir: str = os.path.join(self._base_dir, self.name)
    ### Data :
    self._data_path: str = self.assert_isdir(data_path)
    self._img_path: str = os.path.join(self._data_path, 'imgs')
    self._msks_path: str = os.path.join(self._data_path, 'msks')
    
    self.__seed: int = seed

    ## Create model directory :
    try:
      os.mkdir(self._instance_dir)
    except:
      raise Exception(f"Could not create directory `{self._instance_dir}` at `{self._base_dir}`")

    try:
      def __custom_print_to_summary_file(x):
        with open(self.summary_file, 'a') as f:
          f.write(f"{x}\n")
      self._model.summary(print_fn=__custom_print_to_summary_file)
    except:
      raise Exception(f"Could not dump `model.summary()` output to {self.summary_file}")
    
    ## Create an instance decorator, specifying _log_file_path
    self.__logged: Callable = timing.time_log(self.log_file)

    ## Log initial info to comment file :
    _initial_message = f"\n{self._author}, running '{self._name}' model instantiated at {self._date}\n"
    _initial_message += f"System specifications found in file:\n `{os.path.split(self.hardware_file)[1]}`\n\n"
    with open(self.comment_file, "w") as f:
      f.write(_initial_message)

    ## Log hardware info :
    self.save_hardware_specs()


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
  def name(self) -> str:
    """ Name of the model, composed of: name-(author)-(date) """
    return f"{self._name}-({self._author})-({self._date})"


  @property
  def snapshot_file(self) -> str:
    """ Name (full path) of the model snapshot file (model weights, .h5 format) """
    return os.path.join(self._instance_dir, f"{self.name}-model.h5")


  @property
  def history_file(self) -> str:
    """ Name (full path) of the training history file (.csv format) """
    return os.path.join(self._instance_dir, f"{self.name}-history.csv")


  @property
  def comment_file(self) -> str:
    """ Name (full path) of the comment file (.txt format) """
    return os.path.join(self._instance_dir, f"{self.name}-comments.txt")


  @property
  def log_file(self) -> str:
    """ Name (full path) of the log file (.jsonl format) """
    return os.path.join(self._instance_dir, f"{self.name}-function_calls.jsonl")


  @property
  def summary_file(self) -> str:
    """ Name (full path) of the summary file, 
    yielded by calling self._model.summary() 
    (.txt format) 
    """
    return os.path.join(self._instance_dir, f"{self.name}-summary.txt")


  @property
  def model_file(self) -> str:
    """ Name (full path) of the description of the model (.json format) """
    return os.path.join(self._instance_dir, f"{self.name}-model.json")


  @property 
  def hardware_file(self) -> str:
    """ Name (full path) of the description of the hardware (.jsonl format) """
    return os.path.join(self._instance_dir, f"{self.name}-hardware_specs.jsonl")


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
  def system_info(self) -> Dict[str,str]:
    """ Wrapper for Segmed._getSystemInfo """
    return Segmed._getSystemInfo()

  @property
  def get_gpu(self) -> List[Dict[str,Union[str,float,int]]]:
    """ Wrapper for Segmed._get_gpu """
    return Segmed._get_gpu()


  def save_hardware_specs(self) -> NoReturn:
    """ Save hardware specs to self.hardware_file """
    with open(self.hardware_file, 'w') as f:
      f.write(
          json.dumps({
            "OS/cpu": self.system_info,
            "GPU": self.get_gpu
          })
      )


  def create_train_test_generators(
      self, 
      data_gen_args: Optional[Dict[str, Any]] = None,
      hyper_params: Optional[Dict[str, Any]] =  None
  ) -> NoReturn:
    """ Generate the train/test split. 
    
        Parameters :
            data_gen_args,  defaults to Segmed.__data_gen_args
            hyper_params,   defaults to Segmed.__hyper_params
    """
    self._data_gen_args: Dict[str,Any] = data_gen_args or Segmed.__data_gen_args
    self._hyper_params:  Dict[str,Any] = hyper_params  or Segmed.__hyper_params

    # Crea the training generators with the defined transformations
    self.__image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**self._data_gen_args)
    self.__mask_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(**self._data_gen_args)
    
    # Decoreate the ImageDataGenerator.flow from directory method for logging.
    _img_gen = self.__logged(self.__image_datagen.flow_from_directory)
    # Take images from directories
    self._image_generator_train = _img_gen(
        self._img_path,
        class_mode=None,
        color_mode="rgb",
        batch_size=self._hyper_params["batch_size"],
        seed=self.__seed,
        subset="training",
    )
    
    _msk_gen = self.__logged(self.__mask_datagen.flow_from_directory)
    self._mask_generator_train = _msk_gen(
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
    self._image_generator_val = _img_gen(
        self._img_path,
        class_mode=None,
        color_mode="rgb",
        batch_size=self._hyper_params["batch_size"],
        seed=self.__seed,
        subset="validation",
    )

    self._mask_generator_val = _msk_gen(
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

    
  def create_custom_callback(
      self, 
      model_checkpoint_kw: Optional[Dict[str,Any]] = None
  ) -> NoReturn:
    """ Define the checkpoint callback.
      Parameters (keyword arguments) used to call 
      `tf.keras.callbacks.ModelCheckpoint()` can be 
      specified as a dictionary (parameter `model_checkpoint_kw`).

      Parameters:
        model_checkpoint_kw, defaults to Segmed.__model_checkpoint_kw
    """
    # Define the checkpoint callback, always maximum mode for custom metrics
    self._model_checkpoint_kw: Dict[str,Any] = model_checkpoint_kw or Segmed.__model_checkpoint_kw
      
    self._checkpoint = tf.keras.callbacks.ModelCheckpoint(
        self.snapshot_file, 
        **self._model_checkpoint_kw
    )


  def compile(
      self,
      compiling_kw: Optional[Dict[str,Any]] = None
  ) -> NoReturn:
    """ Compile the model with custom params, creating a log. 
    Arguments:
        compiling_kw (optional), defaults to Segmed.__compiling_kw 
    """ 
    self._compiling_kw: Dict[str,Any] = compiling_kw or Segmed.__compiling_kw 
    ## Decorate model.compile method
    _compile = self.__logged(self._model.compile)
    _compile(**self._compiling_kw)


  def train(
      self, 
      compiling_kw: Dict[str,Any] = None,
      model_checkpoint_kw: Optional[Dict[str,Any]] = None,
      data_gen_args: Optional[Dict[str, Any]] = None,
      hyper_params: Optional[Dict[str, Any]] =  None,
      verbose: bool = True
  ) -> NoReturn:
    """ Train self._model, calling the following methods beforehand :

    self.compile
    self.create_custom_callback
    self.create_train_test_generators

    (See their docstrings)

    Arguments :
        compiling_kw, defaults to Segmed.__compiling_kw
        data_gen_args, defaults to Segmed.__data_gen_args
        hyper_params, defaults to Segmed.__hyper_params 
        model_checkpoint_kw, defaults to Segmed.__model_checkpoint_kw
        verbose, defaults to True
                 this activates printing 
                 some extra info during 
                 the training process.

    Calls a decorated version of `self._model.fit_generator` which 
    will create a log of the parameters passed to it and the execution time.

    Will also save the training history to a csv file :
      self._metrics_history = pd.DataFrame(self._history.history)
      self._metrics_history.to_csv(self.history_file)
    
    """

    self._compiling_kw:  Dict[str,Any] = compiling_kw  or Segmed.__compiling_kw
    self._data_gen_args: Dict[str,Any] = data_gen_args or Segmed.__data_gen_args
    self._hyper_params:  Dict[str,Any] = hyper_params  or Segmed.__hyper_params 
    self._model_checkpoint_kw: Dict[str,Any] = model_checkpoint_kw or Segmed.__model_checkpoint_kw

    self.compile(compiling_kw=compiling_kw)
    self.create_custom_callback(model_checkpoint_kw=model_checkpoint_kw)
    self.create_train_test_generators(data_gen_args=data_gen_args, hyper_params=hyper_params)
    
    # Decorate the model's fit generator to log parameters and execution time :
    _fit_generator = self.__logged(self._model.fit_generator)

    # Create history 
    self._history = _fit_generator(
        self._train_generator,
        callbacks=[self._checkpoint],
        verbose=1,
        validation_data=self._val_generator,
        validation_steps=self._hyper_params["steps_per_epoch"],
        steps_per_epoch=self._hyper_params["steps_per_epoch"],
        epochs=self._hyper_params["epochs"],
        use_multiprocessing=True
    )

    try:
      self._metrics_history = pd.DataFrame(self._history.history)
      self._metrics_history.to_csv(self.history_file)
      print(f"History saved to {self.history_file}")
    except:
      print(f"Could not open file : {self.history_file}")


  def comment(self, cmt: str) -> NoReturn:
    """ Comment something to self.comment_file, logging the Author and UTCdatetime """
    _now = str(datetime.datetime.utcnow()).split(".")[0]
    _cmt = f"\n{self._author} @ {_now} : \n\t{cmt}"
    with open(self.comment_file, 'a') as f:
      f.write(f"{_cmt} \n")

# END Segmed

