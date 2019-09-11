import pytest
import tensorflow as tf
from models import unet


def test_unet_is_model():

    model = unet()

    assert isinstance(model, tf.keras.Model)


def test_unet_classification():

    # TODO: Change the dataset, this one cannot be used for this particular task

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train[:, :, tf.newaxis]
    x_test = x_test[:, :, tf.newaxis]

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    model = unet(input_size=(28, 28, 1))

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.Accuracy()],
    )

    model.fit(
        x=x_train, y=y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test)
    )

    result = model.predict(x_test[0])

    assert result.shape == y_test[0].shape

