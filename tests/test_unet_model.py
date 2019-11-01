import tensorflow.keras as K
from segmed.models import Unet
from segmed.metrics.metrics import jaccard_index
import numpy as np
from . import SimpleDataset


class TestSimpleUnet(SimpleDataset):
    @staticmethod
    def test_simple_unet_is_model():
        """Test that the model is an actual Keras model,
        and that the layers are valid ones.
        """

        model = Unet((256, 256, 3), variant="simple").collect()

        assert isinstance(model, K.Model)

    def test_simple_unet_segmentation(self):
        """Test that the UNet model can train correctly, and that it
        returns a valid result.
        """
        # Import some sample images from within the directory
        x_train, x_test, y_train, y_test = self._create_dataset()
        # Create the model and train it, test the results
        model = Unet(x_train[0].shape, variant="simple").collect()
        model.compile(
            loss=K.losses.BinaryCrossentropy(),
            optimizer=K.optimizers.Adam(),
            metrics=[jaccard_index],
        )
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=1,
            epochs=2,
            validation_data=(x_test, y_test),
        )
        result = model.predict(x_test)

        assert result[0].shape == y_test[0].shape


class TestCustomUnet(SimpleDataset):
    @staticmethod
    def test_custom_unet_is_model():
        """Test that the model is an actual Keras model,
        and that the layers are valid ones.
        """
        conv = {
            "activation": "relu",
            "padding": "same",
            "batch_norm": True,
            "l2_reg": 0.995,
        }
        model = Unet((256, 256, 3), variant="custom", parameters=conv).collect()

        assert isinstance(model, K.Model)

    def test_custom_unet_segmentation(self):
        """Test that the UNet model can train correctly, and that it
        returns a valid result.
        """
        # Import some sample images from within the directory
        x_train, x_test, y_train, y_test = self._create_dataset()
        # Create the model and train it, test the results
        conv = {
            "activation": "relu",
            "padding": "same",
            "batch_norm": True,
            "l2_reg": 0.995,
        }
        model = Unet(x_train[0].shape, variant="custom", parameters=conv).collect()
        model.compile(
            loss=K.losses.BinaryCrossentropy(),
            optimizer=K.optimizers.Adam(),
            metrics=[jaccard_index],
        )
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=1,
            epochs=2,
            validation_data=(x_test, y_test),
        )
        result = model.predict(x_test)

        assert result[0].shape == y_test[0].shape
