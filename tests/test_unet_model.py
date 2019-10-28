import pytest
import tensorflow as tf
from segnet.models import Unet
from segnet.train import train_segnet
import skimage.io as skio
import numpy as np


class TestSimpleUnet:
    def test_simple_unet_is_model(self):
        """Test that the model is an actual Keras model,
        and that the layers are valid ones.
        """

        model = Unet((256, 256, 3), variant="simple").collect()

        assert isinstance(model, tf.keras.Model)

    def test_simple_unet_segmentation(self):
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
        model = Unet(x_train[0].shape, variant="simple").collect()
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.Accuracy()],
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

    # def test_train_unet_with_util(self):
    #     """Test that the UNet model can train correctly, and that it
    #     returns a valid result using the training interface.
    #     """
    #     model = Unet(variant="simple")
    #     result = train_segnet(
    #         model,
    #         "tests/example_dataset/",
    #         "tests/example_dataset/",
    #         batch_size=1,
    #         epochs=2,
    #         val_split=0.0,
    #         optimizer=tf.keras.optimizers.Adam(),
    #         monitor="val_jaccard_index",
    #         model_file="unet_simple.h5",
    #         seed=1,
    #     )


class TestCustomUnet:
    def test_custom_unet_is_model(self):
        """Test that the model is an actual Keras model,
        and that the layers are valid ones.
        """
        conv = {
            "activation": "relu",
            "padding": "same",
            "dropout": 0.5,
            "l2_reg": 0.995,
        }
        model = Unet((256, 256, 3), variant="custom", parameters=conv).collect()

        assert isinstance(model, tf.keras.Model)

    def test_custom_unet_segmentation(self):
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
        conv = {
            "activation": "relu",
            "padding": "same",
            "dropout": 0.5,
            "l2_reg": 0.995,
        }
        model = Unet(x_train[0].shape, variant="custom", parameters=conv).collect()
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.Accuracy()],
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
