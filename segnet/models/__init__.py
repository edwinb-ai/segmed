from .unet import simple_unet
from .segmodel import SegmentationModel
from .multiresunet import *


class Unet(SegmentationModel):
    def __init__(self, input_size, variant=None):
        super().__init__(input_size)

        if variant is "simple":
            self._activation = "relu"
            self._padding = "same"
            self._simple_init()

    def _simple_init(self):

        options = {"activation": self._activation, "padding": self._padding}

        self._seg_model = simple_unet(self._input_size, options)

