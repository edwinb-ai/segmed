from tensorflow import keras as K


def conv2d(x, filters, shape, padding="same", strides=(1, 1), activation="relu"):
    """
    2D Convolutional layers with Batch Normalization
    
    Args:
        x: Keras layer, the input to the feature map
        filters: Int representing the number of filters to use
        shape: Tuple with two integer values (number of rows, number of columns)
        padding: String that determines the padding mode
        strides: Tuple of two integer values that represent the strides
        activation: String that defines the activation function
    
    Returns:
        x: A Keras layer
    """

    x = K.layers.Conv2D(
        filters, shape, strides=strides, padding=padding, use_bias=False
    )(x)
    x = K.layers.BatchNormalization(scale=False)(x)

    if activation is None:
        return x

    x = K.layers.Activation(activation)(x)

    return x


def MultiResBlock(u_val, input, alpha=1.67):
    """MultiRes Block, as defined in the paper.

    Alpha is a constant value that controls
    the number of parameters in the block.
    
    Args:
        U: Integer value for the number of filters.
        input: A Keras layer.
    
    Returns:
        out: A Keras layer.
    """
    # Calculate the value of W as defined in the paper.
    weight = u_val * alpha
    # The first 1x1 map, to preserve dimensions
    dimension_conservation = conv2d(
        input,
        int(weight * 0.167) + int(weight * 0.333) + int(weight * 0.5),
        (1, 1),
        activation=None,
        padding="same",
    )
    # First 3x3 map, adjusted with W / 6
    conv3x3 = conv2d(
        input, int(weight * 0.167), (3, 3), activation="relu", padding="same"
    )
    # Second 3x3 map, adjusted with W / 3
    conv5x5 = conv2d(
        conv3x3, int(weight * 0.333), (3, 3), activation="relu", padding="same"
    )
    # Third 3x3 map, adjusted with W / 2
    conv7x7 = conv2d(
        conv5x5, int(weight * 0.5), (3, 3), activation="relu", padding="same"
    )
    # Concatenate all three 3x3 maps
    out = K.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
    out = K.layers.BatchNormalization()(out)
    # And add the new 7x7 map with the 1x1 map, batch normalized
    out = K.layers.add([dimension_conservation, out])
    out = K.layers.Activation("relu")(out)
    out = K.layers.BatchNormalization()(out)

    return out


def ResPath(filters, input, length=None):
    """ResPath, to mitigate the semantic gap in the architecture.

    This function creates a path with just one combination of residual
    and feature maps, and this can easily be extended with the length
    argument.
    
    Args:
        filters: Integer value corresponding to the number of filters.
        length: Integer value with the length of the path, number of maps.
        input: Keras layer.
    
    Returns:
        out: Keras layer.
    """
    # First residual connection
    residual = conv2d(input, filters, (1, 1), activation=None, padding="same")
    # And first feature map
    out = conv2d(input, filters, (3, 3), activation="relu", padding="same")
    # Add the layers and batch normalize
    out = K.layers.add([residual, out])
    out = K.layers.Activation("relu")(out)
    out = K.layers.BatchNormalization()(out)
    # If there is more maps to add, we add them with this loop
    if not length is None:
        for _ in range(length - 1):

            residual = out
            residual = conv2d(
                residual, filters, (1, 1), activation=None, padding="same"
            )

            out = conv2d(out, filters, (3, 3), activation="relu", padding="same")

            out = K.layers.add([residual, out])
            out = K.layers.Activation("relu")(out)
            out = K.layers.BatchNormalization()(out)

    return out


def MultiResUnet(input_size=(256, 256, 3)):
    """The MultiResUNet neural network.

    A TensorFlow implementation of the MultiResUNet architecture as defined in the
    following paper:
        https://arxiv.org/abs/1902.04049
    
    This is a variant of the U-Net, with additional blocks and paths to help mitigate
    semantic gaps and to obtain better characteristics from the images and maps.
    
    Args:
        input_size: Tuple of three integers (height, width, number of channels) that
            describe the input images.
    
    Returns:
        model: A Keras model instance.
    """

    inputs = K.layers.Input((input_size))

    mresblock_1 = MultiResBlock(32, inputs)
    pool_1 = K.layers.MaxPooling2D(pool_size=(2, 2))(mresblock_1)
    mresblock_1 = ResPath(32, mresblock_1, 4)

    mresblock_2 = MultiResBlock(64, pool_1)
    pool_2 = K.layers.MaxPooling2D(pool_size=(2, 2))(mresblock_2)
    mresblock_2 = ResPath(64, mresblock_2, 3)

    mresblock_3 = MultiResBlock(128, pool_2)
    pool_3 = K.layers.MaxPooling2D(pool_size=(2, 2))(mresblock_3)
    mresblock_3 = ResPath(128, mresblock_3, 2)

    mresblock_4 = MultiResBlock(256, pool_3)
    pool_4 = K.layers.MaxPooling2D(pool_size=(2, 2))(mresblock_4)
    mresblock_4 = ResPath(256, mresblock_4)

    mresblock5 = MultiResBlock(512, pool_4)

    up_6 = K.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(
        mresblock5
    )
    up_6 = K.layers.Concatenate()([up_6, mresblock_4])
    mresblock_6 = MultiResBlock(256, up_6)

    up_7 = K.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(
        mresblock_6
    )
    up_7 = K.layers.Concatenate()([up_7, mresblock_3])
    mresblock7 = MultiResBlock(128, up_7)

    up_8 = K.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(
        mresblock7
    )
    up_8 = K.layers.Concatenate()([up_8, mresblock_2])
    mresblock8 = MultiResBlock(64, up_8)

    up_9 = K.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(
        mresblock8
    )
    up_9 = K.layers.Concatenate()([up_9, mresblock_1])
    mresblock9 = MultiResBlock(32, up_9)

    conv_10 = conv2d(mresblock9, 1, (1, 1), activation="sigmoid")

    model = K.models.Model(inputs=[inputs], outputs=[conv_10])

    return model

