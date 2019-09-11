from tensorflow import keras as K


def unet(input_size=(256, 256, 3)):
    """
    Implementación de U-Net utilizando concatenación en lugar de copiar y cortar
    los mapas de características.
    
    Argumentos:
        input_size: Una tupla de tres elementos que determina el tamaño de la imagen a segmentar.
        Tiene por default (256, 256, 3), i.e. la imagen es de 256x256, con 3 canales de color.

    Regresa:
        model: Una instancia de Model de Keras que sirve para entrenamiento y evaluación.
    """
    entrada = K.Input(input_size)

    # Primer bloque del codificador
    conv_1 = K.layers.Conv2D(64, 3, activation="relu", padding="same")(entrada)
    conv_1 = K.layers.Conv2D(64, 3, activation="relu", padding="same")(conv_1)
    pool_1 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_1)

    # Segundo bloque del codificador
    conv_2 = K.layers.Conv2D(128, 3, activation="relu", padding="same")(pool_1)
    conv_2 = K.layers.Conv2D(128, 3, activation="relu", padding="same")(conv_2)
    pool_2 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

    # Tercer bloque del codificador
    conv_3 = K.layers.Conv2D(256, 3, activation="relu", padding="same")(pool_2)
    conv_3 = K.layers.Conv2D(256, 3, activation="relu", padding="same")(conv_3)
    pool_3 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv_3)

    # Cuarto bloque del codificador
    conv_4 = K.layers.Conv2D(512, 3, activation="relu", padding="same")(pool_3)
    conv_4 = K.layers.Conv2D(512, 3, activation="relu", padding="same")(conv_4)
    drop_4 = K.layers.Dropout(0.5)(conv_4)
    pool_4 = K.layers.MaxPooling2D(pool_size=(2, 2))(drop_4)

    # Conexión, la parte de "hasta abajo" de la U
    conv_5 = K.layers.Conv2D(1024, 3, activation="relu", padding="same")(pool_4)
    conv_5 = K.layers.Conv2D(1024, 3, activation="relu", padding="same")(conv_5)
    drop_5 = K.layers.Dropout(0.5)(conv_5)

    # Primer bloque del decodificador
    up_6 = K.layers.UpSampling2D(size=(2, 2))(drop_5)
    up_6 = K.layers.Conv2D(512, 2, activation="relu", padding="same")(up_6)

    # Concatenación entre el cuarto bloque codificador y el primer bloque decodificador
    merge_6 = K.layers.Concatenate()([conv_4, up_6])
    conv_6 = K.layers.Conv2D(512, 3, activation="relu", padding="same")(merge_6)
    conv_6 = K.layers.Conv2D(512, 3, activation="relu", padding="same")(conv_6)

    # Segundo bloque del decodificador
    up_7 = K.layers.UpSampling2D(size=(2, 2))(conv_6)
    up_7 = K.layers.Conv2D(256, 2, activation="relu", padding="same")(up_7)

    # Concatenación entre el tercer bloque codificador y el segundo bloque decodificador
    merge_7 = K.layers.Concatenate()([conv_3, up_7])
    conv_7 = K.layers.Conv2D(256, 3, activation="relu", padding="same")(merge_7)
    conv_7 = K.layers.Conv2D(256, 3, activation="relu", padding="same")(conv_7)

    # Tercer bloque del decodificador
    up_8 = K.layers.UpSampling2D(size=(2, 2))(conv_7)
    up_8 = K.layers.Conv2D(128, 2, activation="relu", padding="same")(up_8)

    # Concatenación entre el segundo bloque codificador y el tercer bloque decodificador
    merge_8 = K.layers.Concatenate()([conv_2, up_8])
    conv_8 = K.layers.Conv2D(128, 3, activation="relu", padding="same")(merge_8)
    conv_8 = K.layers.Conv2D(128, 3, activation="relu", padding="same")(conv_8)

    # Tercer bloque del decodificador
    up_9 = K.layers.UpSampling2D(size=(2, 2))(conv_8)
    up_9 = K.layers.Conv2D(64, 2, activation="relu", padding="same")(up_9)

    # Concatenación entre el primer bloque codificador y el cuarto bloque decodificador
    merge_9 = K.layers.Concatenate()([conv_1, up_9])
    conv_9 = K.layers.Conv2D(64, 3, activation="relu", padding="same")(merge_9)
    conv_9 = K.layers.Conv2D(64, 3, activation="relu", padding="same")(conv_9)

    # Salida de la U-Net
    conv_9 = K.layers.Conv2D(2, 3, activation="relu", padding="same")(conv_9)
    conv_10 = K.layers.Conv2D(1, 1, activation="sigmoid")(conv_9)

    model = K.Model(inputs=entrada, outputs=conv_10)

    return model
