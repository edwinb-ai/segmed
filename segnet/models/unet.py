def unet(input_size=(256, 256, 3), backend="tf"):
    """
    Implementation of the U-Net model, using Concatenation instead of
    crop and place for the semantic gap.
    
    Arguments:
        input_size: Tuple of three integer values that correspond to the image
            information, namely (height, width, channels).

    Returns:
        model: A tf.keras.Model instance
    """
    # Define the backend for the model
    if backend is "tf":
        from tensorflow import keras as K
    elif backend is "keras":
        import keras as K

    # Take in the inputs
    inputs = K.layers.Input(input_size)

    # First encoder block
    conv_1 = K.layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv_1 = K.layers.Conv2D(64, 3, activation="relu", padding="same")(conv_1)
    pool_1 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_1)

    # Second encoder block
    conv_2 = K.layers.Conv2D(128, 3, activation="relu", padding="same")(pool_1)
    conv_2 = K.layers.Conv2D(128, 3, activation="relu", padding="same")(conv_2)
    pool_2 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

    # Third encoder block
    conv_3 = K.layers.Conv2D(256, 3, activation="relu", padding="same")(pool_2)
    conv_3 = K.layers.Conv2D(256, 3, activation="relu", padding="same")(conv_3)
    pool_3 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_3)

    # Fourth encoder block
    conv_4 = K.layers.Conv2D(512, 3, activation="relu", padding="same")(pool_3)
    conv_4 = K.layers.Conv2D(512, 3, activation="relu", padding="same")(conv_4)
    drop_4 = K.layers.Dropout(0.5)(conv_4)
    pool_4 = K.layers.MaxPooling2D(pool_size=(2, 2))(drop_4)

    # Encoder-decoder conection
    conv_5 = K.layers.Conv2D(1024, 3, activation="relu", padding="same")(pool_4)
    conv_5 = K.layers.Conv2D(1024, 3, activation="relu", padding="same")(conv_5)
    drop_5 = K.layers.Dropout(0.5)(conv_5)

    # First decoder block
    up_6 = K.layers.UpSampling2D(size=(2, 2))(drop_5)
    up_6 = K.layers.Conv2D(512, 2, activation="relu", padding="same")(up_6)

    # Concatenation of first decoder and fourth encoder blocks
    merge_6 = K.layers.Concatenate()([conv_4, up_6])
    conv_6 = K.layers.Conv2D(512, 3, activation="relu", padding="same")(merge_6)
    conv_6 = K.layers.Conv2D(512, 3, activation="relu", padding="same")(conv_6)

    # Second decoder block
    up_7 = K.layers.UpSampling2D(size=(2, 2))(conv_6)
    up_7 = K.layers.Conv2D(256, 2, activation="relu", padding="same")(up_7)

    # Concatenation of second decoder and third encoder block
    merge_7 = K.layers.Concatenate()([conv_3, up_7])
    conv_7 = K.layers.Conv2D(256, 3, activation="relu", padding="same")(merge_7)
    conv_7 = K.layers.Conv2D(256, 3, activation="relu", padding="same")(conv_7)

    # Third decoder block
    up_8 = K.layers.UpSampling2D(size=(2, 2))(conv_7)
    up_8 = K.layers.Conv2D(128, 2, activation="relu", padding="same")(up_8)

    # Concatenation of third decoder and second encoder block
    merge_8 = K.layers.Concatenate()([conv_2, up_8])
    conv_8 = K.layers.Conv2D(128, 3, activation="relu", padding="same")(merge_8)
    conv_8 = K.layers.Conv2D(128, 3, activation="relu", padding="same")(conv_8)

    # Fourth decoder block
    up_9 = K.layers.UpSampling2D(size=(2, 2))(conv_8)
    up_9 = K.layers.Conv2D(64, 2, activation="relu", padding="same")(up_9)

    # Concatenation of fourth decoder and first encoder block
    merge_9 = K.layers.Concatenate()([conv_1, up_9])
    conv_9 = K.layers.Conv2D(64, 3, activation="relu", padding="same")(merge_9)
    conv_9 = K.layers.Conv2D(64, 3, activation="relu", padding="same")(conv_9)

    # Output of U-Net
    conv_9 = K.layers.Conv2D(2, 3, activation="relu", padding="same")(conv_9)
    conv_10 = K.layers.Conv2D(1, 1, activation="sigmoid")(conv_9)

    model = K.models.Model(inputs=[inputs], outputs=[conv_10])

    return model

