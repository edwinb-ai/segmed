import pytest
import tensorflow as tf
from models import unet
import skimage.io as skio
import numpy as np


def test_unet_is_model():

    model = unet()

    assert isinstance(model, tf.keras.Model)


def test_unet_classification():

    x_train = skio.ImageCollection(
        "example_dataset/images_prepped_train/*.png"
    ).concatenate()
    y_train = skio.ImageCollection(
        "example_dataset/annotations_prepped_train/*.png"
    ).concatenate()
    x_test = skio.ImageCollection(
        "example_dataset/images_prepped_test/*.png"
    ).concatenate()
    y_test = skio.ImageCollection(
        "example_dataset/annotations_prepped_test/*.png"
    ).concatenate()

    x_train = x_train[:, :256, :256, :].astype(np.float32)
    y_train = y_train[:, :256, :256, tf.newaxis].astype(np.float32)
    x_test = x_test[:, :256, :256, :].astype(np.float32)
    y_test = y_test[:, :256, :256, tf.newaxis].astype(np.float32)

    x_train /= 255.0
    y_train /= 255.0
    x_test /= 255.0
    y_test /= 255.0

    model = unet(input_size=(256, 256, 3))

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

