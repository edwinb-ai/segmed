from .unet import simple_unet
from .multiresunet import *


class Unet:
    def __init__(self, input_size, variant=None):
        self._input_size = input_size
        self._activation = None
        self._padding = None
        self._pool = None
        self._l1_reg = 0.0
        self._l2_reg = 0.0
        self._l1_l2_reg = 0.0
        self._seg_model = None

        if variant is "simple":
            self._activation = "relu"
            self._padding = "same"
            self._simple_init()

    @property
    def model(self):
        return self._seg_model

    @property
    def input_size(self):
        return self._input_size

    @property
    def activation(self):
        return self._activation

    @property
    def activation(self, activation):
        self._activation = activation

    def _simple_init(self):

        options = {"activation": self._activation, "padding": self._padding}

        self._seg_model = simple_unet(self._input_size, options)

