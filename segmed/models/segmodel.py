class SegmentationModel:
    """A class for a generic segmentation model.

    If one needs to build a segmentation model in Keras, this class
    should provide enough attributes and properties, as well as some
    useful utilities, to make a custom and succint model.

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
    """
    def __init__(self):
        self._input_size = None
        self._filters = 64
        self._kernel_size = 3
        self._dropout = None
        self._batch_norm = None
        self._up_sample = (2, 2)
        self._activation = None
        self._padding = None
        self._pool = (2, 2)
        self._l1_reg = 0.0
        self._l2_reg = 0.0
        self._seg_model = None

    @property
    def model(self):
        return self._seg_model

    @property
    def filters(self):
        return self._filters

    @filters.setter
    def filters(self, value):
        self._filters = value

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        self._kernel_size = value

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, value):
        self._dropout = value

    @property
    def batch_norm(self):
        return self._batch_norm

    @batch_norm.setter
    def batch_norm(self, value):
        self._batch_norm = value

    @property
    def up_sample(self):
        return self._up_sample

    @up_sample.setter
    def up_sample(self, value):
        self._up_sample = value

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value):
        self._activation = value

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, value):
        self._padding = value

    @property
    def pool(self):
        return self._pool

    @pool.setter
    def pool(self, value):
        self._pool = value

    @property
    def l1_reg(self):
        return self._l1_reg

    @l1_reg.setter
    def l1_reg(self, value):
        self._l1_reg = value

    @property
    def l2_reg(self):
        return self._l2_reg

    @l2_reg.setter
    def l2_reg(self, value):
        self._l2_reg = value

    @property
    def l1_l2_reg(self):
        return self._l1_l2_reg

    @l1_l2_reg.setter
    def l1_l2_reg(self, value):
        self._l1_l2_reg = value

    def _parse_params(self, params):
        """
        Read a dictionary of parameters and store them in the
        class attributes.

        Args:
            params (dict): A dictionary of attributes. Read the class
                docstrings to know more.
        """
        if "l1_reg" in params:
            self._l1_reg = params["l1_reg"]
        if "l2_reg" in params:
            self._l2_reg = params["l2_reg"]
        if "activation" in params:
            self._activation = params["activation"]
        if "filters" in params:
            self._filters = params["filters"]
        if "kernel_size" in params:
            self._kernel_size = params["kernel_size"]
        if "pool" in params:
            self._pool = params["pool"]
        if "padding" in params:
            self._padding = params["padding"]
        if "dropout" in params:
            self._dropout = params["dropout"]
        if "up_sample" in params:
            self._up_sample = params["up_sample"]
