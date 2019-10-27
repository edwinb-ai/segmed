import pytest
import tensorflow as tf
from segnet.models.unet import Unet
import skimage.io as skio
import numpy as np


class TestSimpleUnet:
    def test_simple_unet_is_model(self):
        """Test that the model is an actual Keras model,
        and that the layers are valid ones.
        """

        model = Unet((256, 256, 3), variant="simple").model

        assert isinstance(model, tf.keras.Model)


    def test_unet_segmentation(self):
        """Test that the UNet model can train correctly, and that it
        returns a valid result.
        """
        # Import some sample images from within the directory
        x_train = skio.ImageCollection(
            "tests/example_dataset/images_prepped_train/*.png"
        ).concatenate()
        y_train = skio.ImageCollection(
            "tests/example_dataset/annotations_prepped_train/*.png"
        ).concatenate()
        x_test = skio.ImageCollection(
            "tests/example_dataset/images_prepped_test/*.png"
        ).concatenate()
        y_test = skio.ImageCollection(
            "tests/example_dataset/annotations_prepped_test/*.png"
        ).concatenate()
        # Crop the images to 256x256 and convert them to float32
        x_train = x_train[:, :256, :256, :].astype(np.float32)
        y_train = y_train[:, :256, :256, None].astype(np.float32)
        x_test = x_test[:, :256, :256, :].astype(np.float32)
        y_test = y_test[:, :256, :256, None].astype(np.float32)
        # Normalize the images
        x_train /= 255.0
        y_train /= 255.0
        x_test /= 255.0
        y_test /= 255.0
        # Create the model and train it, test the results
        model = Unet((256, 256, 3), variant="simple").model
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.Accuracy()],
        )
        model.fit(
            x=x_train, y=y_train, batch_size=1, epochs=2, validation_data=(x_test, y_test)
        )
        result = model.predict(x_test)

        assert result[0].shape == y_test[0].shape
