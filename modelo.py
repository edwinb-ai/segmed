from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Reshape
from keras.layers import ZeroPadding2D, BatchNormalization, UpSampling2D
from keras.layers import Activation
from keras.utils import get_file


def vgg16_encoder(input_height=128, input_width=128, pretrained="imagenet"):

    pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

    # Considerando que la imagen está en escala de grises
    img_input = Input(shape=(input_height, input_width, 1))

    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(
        img_input
    )
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
    f5 = x

    if pretrained == "imagenet":
        VGG_Weights_path = get_file(pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(VGG_Weights_path)

    return img_input, [f1, f2, f3, f4, f5]


def vgg16_unet(n_classes, input_height=128, input_width=128):

    img_input, levels = vgg16_encoder(
        input_height=input_height, input_width=input_width
    )
    [f1, f2, f3, f4, f5] = levels

    o = f5
    # Capa que une, la de hasta abajo
    # o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding="same"))(o)
    o = (BatchNormalization())(o)
    # 
    o = (UpSampling2D((2, 2)))(o)
    o = concatenate([o, f5], axis=-1)
    # o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding="same"))(o)
    o = (Conv2D(512, (3, 3), padding="same"))(o)
    o = (Conv2D(512, (3, 3), padding="same"))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = concatenate([o, f4], axis=-1)
    # o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding="same"))(o)
    o = (Conv2D(512, (3, 3), padding="same"))(o)
    o = (Conv2D(512, (3, 3), padding="same"))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = concatenate([o, f3], axis=-1)
    # o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding="same"))(o)
    o = (Conv2D(256, (3, 3), padding="same"))(o)
    o = (Conv2D(256, (3, 3), padding="same"))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = concatenate([o, f2], axis=-1)
    # o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding="same"))(o)
    o = (Conv2D(128, (3, 3), padding="same"))(o)
    o = (Conv2D(128, (3, 3), padding="same"))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = concatenate([o, f1], axis=-1)
    # o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding="same"))(o)
    o = (Conv2D(64, (3, 3), padding="same"))(o)
    o = (Conv2D(64, (3, 3), padding="same"))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding="same")(o)
    o = (Activation("sigmoid"))(o)
    o = (Reshape((input_height * input_width, -1)))(o)

    modelo_final = Model(img_input, o)

    return modelo_final
