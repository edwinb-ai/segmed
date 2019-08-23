from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import modelo
import metricas as mets
import preprocesamiento as prep
import numpy as np


# Extraer las imágenes y normalizar
X_train, y_train = prep.extraer_datos(
    "dataset/train_images/train-volume.tif", "dataset/train_images/train-labels.tif"
)
# Extracción de pedazos de imagen
x_patches, y_patches = prep.imagen_en_partes(
    X_train[0], y_train[0], size=(128, 128), num_partes=4
)

# Hacer aumento de datos simple
datagen_x = ImageDataGenerator(
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=75,
    # randomly shift images horizontally
    width_shift_range=0.2,
    # randomly shift images vertically
    height_shift_range=0.2,
    # set range for random shear
    shear_range=0.2,
    # set range for random zoom
    zoom_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    validation_split=0.15,
)
datagen_x.fit(x_patches)

# Crear callbacks
guardar_modelo = ModelCheckpoint("unet.h5", verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=1)

# Compilar el modelo para dos clases, vgg16_unet
# modelo_vgg16_unet = modelo.vgg16_unet(2)
# modelo_vgg16_unet.compile(
#     loss=mets.ternaus_loss,
#     optimizer=SGD(1e-2, momentum=0.9),
#     metrics=[mets.indice_jaccard],
# )
# U-net
modelo_unet = modelo.unet(input_size=(128, 128, 1))
modelo_unet.compile(
    loss=mets.ternaus_loss,
    optimizer=SGD(1e-2, momentum=0.9),
    metrics=[mets.indice_jaccard],
)

# Algunos hiperparámetros
epocas = 10
tam_lote = 2

# Entrenar con generadores
# historia = modelo_vgg16_unet.fit_generator(
#     datagen_x.flow(
#         x_patches, y_patches, batch_size=tam_lote, subset="training"
#     ),
#     epochs=epocas,
#     validation_data=datagen_x.flow(
#         x_patches, y_patches, batch_size=tam_lote, subset="validation"
#     ),
#     steps_per_epoch=10,
#     validation_steps=10,
#     verbose=1,
#     callbacks=[guardar_modelo, reduce_lr],
# )

print(modelo_unet.summary())

historia = modelo_unet.fit(
    x_patches,
    y_patches,
    epochs=epocas,
    validation_split=0.1,
    verbose=1,
    callbacks=[guardar_modelo, reduce_lr],
)

