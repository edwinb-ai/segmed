from tensorflow.keras.regularizers import L1L2
from .unet import simple_unet, custom_unet
from .segmodel import SegmentationModel
from .multiresunet import *


class Unet(SegmentationModel):
    """Unified interface for the simple and custom versions of the UNet.

    Attributes:
        _input_size (Tuple[int, int, int]): Size of the image, (height, width, channels)
        _filters (int): the number of output filters in the convolution
        _kernel_size (int): specifying the length of the 2D convolution window
        _dropout (float): Value for the dropout layer, between 0 and 1.
        _batch_norm (bool): Add batch normalization to every encoder block.
        _up_sample (Tuple[int, int]): Upsampling factor for rows and columns.
        _activation (str): Activation function to apply to every layer, except the last one.
        _padding (str): Type of padding to apply to the convolution layers.
        _pool (Tuple[int, int]): Pooling windows
        _l1_reg (float): Value for the L1 regularizer, applied to every convolution map
        _l2_reg (float): Value for the L2 regularizer, applied to every convolution map
        _seg_model (keras.Model): A tf.keras.Model instance.
        _from_keras (dict): Additional parameters from the Keras API than can apply ONLY to
            convolutional layers.
    """

    def __init__(
        self,
        input_size=(512, 512, 3),
        variant=None,
        parameters=None,
        keras_parameters=None,
    ):
        """Creates a Unet model with the specified variant.

        For the custom version, almost ALL parameters must be specified.

        Args:
            input_size (Tuple[int, int, int]): Size of the image, (height, width, channels)
            variant (str): Either "simple" or "custom", specifying the type of model
            parameters (dict): All the parameters to add from the SegmentationModel specification.
                See the documentation for more on the attributes.
            keras_parameters (dict): Additional parameters from the Keras API than can apply ONLY to
                convolutional layers.
        """
        super().__init__()
        self._input_size = input_size
        self._from_keras = keras_parameters
        self._variant = variant

        # The simple model has a ReLU activation and "same" padding
        if self._variant is "simple":
            self._activation = "relu"
            self._padding = "same"
        # For the custom model, parse all the parameters from the dictionary
        if self._variant is "custom":
            if parameters is None:
                raise ValueError("For a custom network, parameters must be set.")
            # Create the model with the specified parameters
            self._parse_params(parameters)

    def _simple_init(self):
        # Build the SegmentationModel API dictionary of parameters
        options = {"activation": self._activation, "padding": self._padding}
        self._seg_model = simple_unet(self._input_size, options)

    def _custom_init(self):
        # Build the Keras API dictionary of parameters
        conv = {
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "activation": self._activation,
            "padding": self._padding,
            "kernel_regularizer": L1L2(l1=self._l1_reg, l2=self._l2_reg),
        }
        # Add additional Keras parameters if passed
        if self._from_keras is not None:
            conv = {**conv, **self._from_keras}
        # Build the SegmentationModel API dictionary of parameters
        options = {
            "pool": self._pool,
            "dropout": self._dropout,
            "batch_norm": self._batch_norm,
            "up_sample": self._up_sample,
        }
        self._seg_model = custom_unet(self._input_size, conv, **options)

    def collect(self):
        """Gather all the information from the model, inside and outside.

        Generate a Keras instance from all the information.
        This way the model can be instantiated
        from the beginning or by updating the attributes individually until you are
        satisfied with the model you have.

        Returns:
            _seg_model (keras.Model): A tf.keras.Model instance with all the information
                from the attributes.
        """
        if self._variant is "simple":
            self._simple_init()

        if self._variant is "custom":
            self._custom_init()

        return self._seg_model
