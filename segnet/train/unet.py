from segnet.models import unet
from segnet.metrics import metrics as mts
import tensorflow as tf
import keras as K


def train_unet(
    img_path,
    mask_path,
    batch_size=16,
    epochs=25,
    steps_per_epoch=3125,
    model_file="unet_simple.h5",
    show=False,
):

    # Instanciar el modelo de la UNet
    model = unet()

    # Definir las transformaciones, reescalar y el formato
    data_gen_args = {"rescale": 1.0 / 255.0, "dtype": tf.float32}

    # Crear los generadores de imágenes y máscaras
    image_datagen = K.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_datagen = K.preprocessing.image.ImageDataGenerator(**data_gen_args)

    # Dejar fija la semilla por ahora
    seed = 1

    # A partir del directorio agarrar las imágenes y máscaras
    image_generator = image_datagen.flow_from_directory(
        img_path, class_mode=None, color_mode="rgb", batch_size=batch_size, seed=seed
    )
    mask_generator = mask_datagen.flow_from_directory(
        mask_path,
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        seed=seed,
    )

    # Combinar ambos en un solo generador
    train_generator = zip(image_generator, mask_generator)

    # Cuando no se está seguro, se pueden ver las imágenes de esta forma
    # SOLAMENTE PARA DEPURACIÓN
    if show:
        import matplotlib.pyplot as plt

        for i, j in train_generator:
            plt.figure(0)
            plt.imshow(i[0, ...])
            plt.figure(1)
            plt.imshow(j[0, ..., 0])
            plt.show()

    # Definir el punto de guardado para cada modelo
    checkpoint = K.callbacks.ModelCheckpoint(
        model_file, monitor="jaccard_index", verbose=1, save_best_only=True, mode="max"
    )

    # Compilar el modelo con las métricas necesarias y un optimizador base
    model.compile(
        loss="binary_crossentropy",
        optimizer="Adam",
        metrics=[mts.jaccard_index, mts.dice_coef],
    )

    # Crear la historia del modelo
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[checkpoint],
        verbose=1,
    )

    return history
