class SegmentationModel:
    def __init__(self, input_size):
        self._input_size = input_size
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
        self._l1_l2_reg = 0.0
        self._seg_model = None

    @property
    def model(self):
        return self._seg_model

    # TODO: Add properties for
    # - filters
    # - kernel_size
    # - droput
    # - batch_nrom
    # - up_sample
    

    @property
    def input_size(self):
        return self._input_size

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, activation):
        self._activation = activation

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
        if "l1_reg" in params:
            self._l1_reg = params["l1_reg"]
        if "l2_reg" in params:
            self._l2_reg = params["l2_reg"]
        if "l1_l2_reg" in params:
            self._l1_l2_reg = params["l1_l2_reg"]
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
