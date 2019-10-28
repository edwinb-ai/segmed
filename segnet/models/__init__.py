from .unet import simple_unet, custom_unet
from .segmodel import SegmentationModel
from .multiresunet import *


class Unet(SegmentationModel):
    def __init__(self, input_size, variant=None, parameters=None):
        super().__init__(input_size)

        if variant is "simple":
            self._activation = "relu"
            self._padding = "same"
            self._simple_init()

        if variant is "custom":
            if parameters is None:
                raise ValueError("For a custom network, parameters must be set.")
            # Create the model with the specified parameters
            self._parse_params(parameters)
            self._custom_init()

    def _simple_init(self):

        options = {"activation": self._activation, "padding": self._padding}

        self._seg_model = simple_unet(self._input_size, options)

    def _custom_init(self):
        conv = {
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "activation": self._activation,
            "padding": self._padding,
        }
        options = {
            "pool": self._pool,
            "dropout": self._dropout,
            "batch_norm": self._batch_norm,
            "up_sample": self._up_sample,
        }
        self._seg_model = custom_unet(self._input_size, conv, **options)
