.. _user-guide:

User guide
===========

Demo
-----

The following is a simple demo of how ``segmed`` works:

.. code-block :: python

    from segmed.train import train_unet
    from segmed.models import Unet

    # Define some example hyperparameters
    batch_size = 8
    epochs = 50
    steps_per_epoch=100

    # Declare the paths to use (following the Keras convention)
    # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#fit_generator
    data_path = "general/path/to/images"
    img_path = data_path + "augmented/images/path"
    masks_path = data_path + "augmented/masks/path"
    model_file = "path/to/save/model/unet_model.h5"

    # Create a Unet (custom) model with a regularizer and
    # batch normalization
    custom_params = {
        "activation": "relu",
        "padding": "same",
        "batch_norm": True,
        "l2_reg": 0.995
    }
    model = Unet(variant="custom", parameters=custom_params)
    # Train the model!
    history = train_unet(
        model,
        img_path,
        masks_path,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        model_file=model_file,
    )


Creating a model
-----------------
