from tensorflow import keras as K
from warnings import warn


def simple_unet(input_size, conv):
    """
    Implementation of the U-Net model, using Concatenation instead of
    crop and place for the semantic gap.
    
    Args:
        input_size (Tuple[int, int, int]): (height, width, channels), i.e.
        The information of the image.
        Must always be a multiple of 32, e.g. 256, 512.
        conv (dict): Keyword parameters from the Keras specification. This dictionary
            will affect ALL convolutional layers in the network.

    Returns:
        model: A tf.keras.Model instance.
    """

    # Take in the inputs
    inputs = K.layers.Input(input_size)

    # First encoder block
    conv_1 = K.layers.Conv2D(64, 3, **conv)(inputs)
    conv_1 = K.layers.Conv2D(64, 3, **conv)(conv_1)
    pool_1 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_1)

    # Second encoder block
    conv_2 = K.layers.Conv2D(128, 3, **conv)(pool_1)
    conv_2 = K.layers.Conv2D(128, 3, **conv)(conv_2)
    pool_2 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

    # Third encoder block
    conv_3 = K.layers.Conv2D(256, 3, **conv)(pool_2)
    conv_3 = K.layers.Conv2D(256, 3, **conv)(conv_3)
    pool_3 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_3)

    # Fourth encoder block
    conv_4 = K.layers.Conv2D(512, 3, **conv)(pool_3)
    conv_4 = K.layers.Conv2D(512, 3, **conv)(conv_4)
    # drop_4 = K.layers.Dropout(0.5)(conv_4)
    pool_4 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_4)

    # Encoder-decoder conection
    conv_5 = K.layers.Conv2D(1024, 3, **conv)(pool_4)
    conv_5 = K.layers.Conv2D(1024, 3, **conv)(conv_5)
    # drop_5 = K.layers.Dropout(0.5)(conv_5)

    # First decoder block
    up_6 = K.layers.UpSampling2D(size=(2, 2))(conv_5)
    up_6 = K.layers.Conv2D(512, 2, **conv)(up_6)

    # Concatenation of first decoder and fourth encoder blocks
    merge_6 = K.layers.Concatenate()([conv_4, up_6])
    conv_6 = K.layers.Conv2D(512, 3, **conv)(merge_6)
    conv_6 = K.layers.Conv2D(512, 3, **conv)(conv_6)

    # Second decoder block
    up_7 = K.layers.UpSampling2D(size=(2, 2))(conv_6)
    up_7 = K.layers.Conv2D(256, 2, **conv)(up_7)

    # Concatenation of second decoder and third encoder block
    merge_7 = K.layers.Concatenate()([conv_3, up_7])
    conv_7 = K.layers.Conv2D(256, 3, **conv)(merge_7)
    conv_7 = K.layers.Conv2D(256, 3, **conv)(conv_7)

    # Third decoder block
    up_8 = K.layers.UpSampling2D(size=(2, 2))(conv_7)
    up_8 = K.layers.Conv2D(128, 2, **conv)(up_8)

    # Concatenation of third decoder and second encoder block
    merge_8 = K.layers.Concatenate()([conv_2, up_8])
    conv_8 = K.layers.Conv2D(128, 3, **conv)(merge_8)
    conv_8 = K.layers.Conv2D(128, 3, **conv)(conv_8)

    # Fourth decoder block
    up_9 = K.layers.UpSampling2D(size=(2, 2))(conv_8)
    up_9 = K.layers.Conv2D(64, 2, **conv)(up_9)

    # Concatenation of fourth decoder and first encoder block
    merge_9 = K.layers.Concatenate()([conv_1, up_9])
    conv_9 = K.layers.Conv2D(64, 3, **conv)(merge_9)
    conv_9 = K.layers.Conv2D(64, 3, **conv)(conv_9)

    # Output of U-Net
    conv_9 = K.layers.Conv2D(2, 3, **conv)(conv_9)
    conv_10 = K.layers.Conv2D(1, 1, activation="sigmoid")(conv_9)

    model = K.models.Model(inputs=[inputs], outputs=[conv_10], name="unet_simple")

    return model


def _encoder(x, conv, dropout=None, batch_norm=False):
    """
    Create an encoder block with dropout, batch normalization, kernel regularatization and
    more. It basically supports every possible parameter from the Keras API.

    Args:
        x (keras.Layer): Layer to build upon.
        conv (dict): All possible arguments from the Keras specification.
        dropout (float): Value for the dropout layer, between 0 and 1.
        batc_norm (bool): Whether to add batch normalization or not.

    Returns:
        some_layer (keras.Layer): Output layer with convolutions and additional
            parameters and layers.
    """
    if batch_norm:
        if dropout is not None:
            warn("Is it NOT recommended to use Batch Normalization with Dropout")

        conv["use_bias"] = False

        some_layer = K.layers.Conv2D(**conv)(x)
        some_layer = K.layers.Conv2D(**conv)(some_layer)
        some_layer = K.layers.BatchNormalization()

    else:
        some_layer = K.layers.Conv2D(**conv)(x)
        some_layer = K.layers.Conv2D(**conv)(some_layer)

    if dropout is not None:
        if dropout <= 0.0:
            raise ValueError("Dropout must be larger than zero.")
        some_layer = K.layers.Dropout(dropout)(some_layer)

    return some_layer


def _concatenate_and_upsample(prev_layer, cat_layer, upsample, conv):
    """
    Decoder blocks for the Unet where upsampling and concatenation take part.

    Args:
        prev_layer (keras.Layer): Previous layer from the network.
        cat_layer (keras.Layer): The concatenation layer that will be merged together
            with the upsampling one.
        upsample (Tuple[int, int]): Upsampling factor for rows and columns.
        conv (dict): All possible arguments from the Keras specification.

    Returns:
        output (keras.Layer): Result from the upsampling and concatenation.
    """
    some_layer = K.layers.UpSampling2D(size=upsample)(prev_layer)
    conv["kernel_size"] = 2
    some_layer = K.layers.Conv2D(**conv)(some_layer)

    merge_layer = K.layers.Concatenate()([cat_layer, some_layer])
    conv["kernel_size"] = 3
    output = K.layers.Conv2D(**conv)(merge_layer)
    output = K.layers.Conv2D(**conv)(output)

    return output


def custom_unet(
    input_size, conv, pool=None, dropout=None, batch_norm=None, up_sample=None
):
    """
    Implementation of the U-Net model but with extreme flexibility to add new paramaters
    like dropout, batch normalization, kernel regularizers and so on.
    
    Args:
        input_size (Tuple[int, int, int]): (height, width, channels), i.e.
        The information of the image.
        Must always be a multiple of 32, e.g. 256, 512.
        conv (dict): Keyword parameters from the Keras specification. This dictionary
            will affect ALL convolutional layers in the network.
        pool (Tuple[int, int]): Pooling windows
        dropout (float): Value for the dropout layer, between 0 and 1.
        batch_norm (bool): Add batch normalization to every encoder block.
        up_sample (Tuple[int, int]): Upsampling factor for rows and columns.
    Returns:
        model: A tf.keras.Model instance.
    """
    inputs = K.layers.Input(input_size)
    encoder_block = inputs
    encoding_layers = []

    for i in range(4):
        encoder_block = _encoder(
            encoder_block, conv, dropout=dropout, batch_norm=batch_norm
        )
        #
        encoding_layers.append(encoder_block)
        # Add the necessary pooling
        if pool is None:
            # Default pooling
            pool = (2, 2)
        encoder_block = K.layers.MaxPooling2D(pool_size=pool)(encoder_block)
        # Update filter size
        conv["filters"] *= 2

    # Add encoder-decoder block
    output_layer = _encoder(encoder_block, conv, dropout=dropout, batch_norm=batch_norm)

    for layer in reversed(encoding_layers):
        # Reduce number of filters
        conv["filters"] //= 2
        if up_sample is None:
            up_sample = (2, 2)
        output_layer = _concatenate_and_upsample(output_layer, layer, up_sample, conv)

    # Output of U-Net
    output_layer = K.layers.Conv2D(
        2, 3, activation=conv["activation"], padding=conv["padding"]
    )(output_layer)
    output_layer = K.layers.Conv2D(1, 1, activation="sigmoid")(output_layer)

    model = K.models.Model(inputs=[inputs], outputs=[output_layer], name="unet_custom")

    return model

