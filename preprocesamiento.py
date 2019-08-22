from skimage import io
from sklearn.feature_extraction import image
import numpy as np
import matplotlib.pyplot as plt


def extraer_datos(train_path, label_path, rgb=False, show=False):
    # Definir el camino de las imágenes
    path_imagenes_entrenamiento = train_path
    path_etiquetas_entrenamiento = label_path
    # Importar las imágenes en formato tif
    X_train = io.imread(path_imagenes_entrenamiento, plugin="tifffile")
    y_train = io.imread(path_etiquetas_entrenamiento, plugin="tifffile")
    # Reacomodar el arreglo para que tenga canales de color
    if rgb:
        # Se agregan los tres canales
        nuevo_tam = list(X_train.shape) + [3]
    else:
        # Solamente se agrega uno, blanco y negro
        nuevo_tam = list(X_train.shape) + [1]
    # Con estos tamaños, reajustar las imágenes
    X_train = X_train.reshape(tuple(nuevo_tam))
    y_train = y_train.reshape(tuple(nuevo_tam))
    # Mostrar una imagen
    if show:
        rand_entero = np.random.randint(0, len(X_train))
        # Mostrar siempre su mapa de segmentación
        list_imgs = [X_train[rand_entero, :, :, 0], y_train[rand_entero, :, :, 0]]
        for i in list_imgs:
            plt.figure()
            io.imshow(i)
        # Devolver el índice de la imagen vista para darle seguimiento
        return X_train, y_train, rand_entero

    return X_train, y_train


def imagen_en_partes(x, y, size=(128, 128), num_partes=4):
    """
    Asumiendo que la imagen es (ancho, alto, canales)
    """
    x_patches = image.extract_patches_2d(
        x[:, :, 0], patch_size=size, max_patches=num_partes, random_state=0
    )
    y_patches = image.extract_patches_2d(
        y[:, :, 0], patch_size=size, max_patches=num_partes, random_state=0
    )

    return x_patches, y_patches


if __name__ == "__main__":
    imagenes_entrenamiento = "dataset/train_images/train-volume.tif"
    etiquetas_entrenamiento = "dataset/train_images/train-labels.tif"
    X, y, imagen_vista = extraer_datos(
        imagenes_entrenamiento, etiquetas_entrenamiento, show=True
    )
    # Extracción de pedazos de imagen
    x_patches, y_patches = imagen_en_partes(
        X[imagen_vista], y[imagen_vista], size=(64, 64), num_partes=8
    )
    for i, j in zip(x_patches, y_patches):
        plt.figure()
        io.imshow(i)
        plt.figure()
        io.imshow(j)
    io.show()
