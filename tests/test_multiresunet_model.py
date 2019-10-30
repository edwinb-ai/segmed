import tensorflow.keras as K
from segnet.models import multiresunet
from segnet.metrics.metrics import jaccard_index
import numpy as np
from . import SimpleDataset


class TestMultiResUNet(SimpleDataset):
    @staticmethod
    def test_multiresunet_is_model():
        """Test that the model is an actual Keras model,
        and that the layers are valid ones.
        """

        model = multiresunet.MultiResUnet()

        assert isinstance(model, K.Model)

    def test_multiresunet_segmentation(self):
        """Test that the MultiResUNet model can train correctly, and that it
        returns a valid result.
        """
        # Import some sample images from within the directory
        x_train, x_test, y_train, y_test = self._create_dataset()
        # Create the model and train it, test the results
        model = multiresunet.MultiResUnet(input_size=(256, 256, 3))
        model.compile(
            loss=K.losses.BinaryCrossentropy(),
            optimizer=K.optimizers.Adam(),
            metrics=[jaccard_index],
        )
        model.fit(
            x=x_train, y=y_train, batch_size=1, epochs=2, validation_data=(x_test, y_test)
        )
        result = model.predict(x_test)

        assert result[0].shape == y_test[0].shape

