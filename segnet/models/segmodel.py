class SegmentationModel:
    def __init__(self, input_size):
        self._input_size = input_size
        self._activation = None
        self._padding = None
        self._pool = None
        self._l1_reg = 0.0
        self._l2_reg = 0.0
        self._l1_l2_reg = 0.0
        self._seg_model = None

    @property
    def model(self):
        return self._seg_model

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
    def pool_size(self):
        return self._pool

    @pool_size.setter
    def pool_size(self, value):
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
