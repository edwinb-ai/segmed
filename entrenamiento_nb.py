#%%
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import modelo
import metricas as mets
import preprocesamiento as prep
import numpy as np


#%%
# Algunos hiperparámetros
epocas = 15
tam_lote = 2
#%%
# Extraer las imágenes y normalizar
X_train, y_train = prep.extraer_datos(
    "dataset/train_images/train-volume.tif", "dataset/train_images/train-labels.tif"
)
#%%
x_patches, y_patches = prep.muchas_imagenes_en_partes(
    X_train, y_train, size=(128, 128), num_partes=4
)
#%%
# Hacer aumento de datos simple
transformaciones = dict(
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=10.0,
    # randomly shift images horizontally
    width_shift_range=0.02,
    # randomly shift images vertically
    height_shift_range=0.02,
    # set range for random shear
    shear_range=5,
    # set range for random zoom
    zoom_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.2
)
#%%
datagen_x = ImageDataGenerator(**transformaciones)
datagen_x.fit(x_patches)
datagen_y = ImageDataGenerator(**transformaciones)
datagen_y.fit(y_patches)
#%%
# Conjunto de entrenamiento
X_train_aumentado = datagen_x.flow(x_patches, batch_size=tam_lote, subset="training")
y_train_aumentado = datagen_y.flow(y_patches, batch_size=tam_lote, subset="training")
entrenamiento = zip(X_train_aumentado, y_train_aumentado)
# Conjunto de validacion
X_train_validacion = datagen_x.flow(x_patches, batch_size=tam_lote, subset="validation")
y_train_validacion = datagen_y.flow(y_patches, batch_size=tam_lote, subset="validation")
validacion = zip(X_train_validacion, y_train_validacion)
#%%
# Crear callbacks
guardar_modelo = ModelCheckpoint("unet.h5", verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=1)

#%%
# U-net
modelo_unet = modelo.unet(input_size=(128, 128, 1))
modelo_unet.compile(
    loss=mets.ternaus_loss,
    optimizer=SGD(1e-2, momentum=0.9),
    metrics=[mets.indice_jaccard],
)
#%%
print(modelo_unet.summary())
#%%
historia = modelo_unet.fit_generator(
    entrenamiento,
    epochs=epocas,
    steps_per_epoch=np.ceil(len(x_patches) // tam_lote).astype("int32"),
    validation_data=validacion,
    validation_steps=np.ceil(len(x_patches) // tam_lote).astype("int32"),
    verbose=1,
    callbacks=[guardar_modelo, reduce_lr]
)
