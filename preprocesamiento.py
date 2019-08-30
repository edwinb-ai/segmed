from skimage import io
from sklearn.feature_extraction import image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


def extraer_datos(train_path, label_path, rgb=False, show=False, tiff=True):
    """
    Extrae los datos de train_path y label_path y los normaliza.

    Argumentos:
        train_path: String que tiene el camino completo del directorio donde están las imágenes.
        label_path: String que tiene el camino completo del directorio donde están los mapas de segmentación.
        rgb: Booleano para reestructurar a forma de las imágenes si son de color o grises.
        show: Booleano, si es verdadero muestra una imagen aleatoria de las que se están importando.
        tiff: Booleando, si es verdadero es porque las imágenes están en formato .tif.

    Regresa:
        X_train, y_train: Arreglos de numpy que contienen las imágenes importadas.
    """
    # Definir el camino de las imágenes
    path_imagenes_entrenamiento = train_path
    path_etiquetas_entrenamiento = label_path

    # Importar las imágenes en formato tif
    if tiff:
        X_train = io.imread(path_imagenes_entrenamiento, plugin="tifffile")
        y_train = io.imread(path_etiquetas_entrenamiento, plugin="tifffile")
    else:
        X_train = io.imread(path_imagenes_entrenamiento)
        y_train = io.imread(path_etiquetas_entrenamiento)
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
    
    # Convertir y normalizar las imágenes
    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32")
    X_train /= 255
    y_train /= 255

    return X_train, y_train

def datos_prueba(test_path, rgb=False, show=False, tiff=True):
    """
    Importa los datos de test_path y los normaliza.

    Argumentos:
        test_path: String que tiene el camino completo del directorio donde están las imágenes.
        rgb: Booleano para reestructurar a forma de las imágenes si son de color o grises.
        show: Booleano, si es verdadero muestra una imagen aleatoria de las que se están importando.
        tiff: Booleando, si es verdadero es porque las imágenes están en formato .tif.

    Regresa:
        X_test: Arreglo de numpy que contiene las imágenes importadas.
    """
    if tiff:
        # Importar las imágenes en formato tif
        X_test = io.imread(test_path, plugin="tifffile")
    else:
        X_test = io.imread(test_path)

    # Reacomodar el arreglo para que tenga canales de color
    if rgb:
        # Se agregan los tres canales
        nuevo_tam = list(X_test.shape) + [3]
    else:
        # Solamente se agrega uno, blanco y negro
        nuevo_tam = list(X_test.shape) + [1]
    # Con estos tamaños, reajustar las imágenes
    X_test = X_test.reshape(tuple(nuevo_tam))
    # Mostrar una imagen
    if show:
        rand_entero = np.random.randint(0, len(X_test))
        # Mostrar siempre su mapa de segmentación
        imagen_prueba = X_test[rand_entero, :, :, 0]
        plt.figure()
        io.imshow(imagen_prueba)
    
    # Convertir y normalizar las imágenes
    X_test = X_test.astype("float32")
    X_test /= 255

    return X_test

def muchas_imagenes_en_partes(x, y=None, size=(128, 128), num_partes=4, rgb=True):
    """
    Toma dos arreglos de imágenes, x,y, y las separa en num_partes número de imágenes.

    Argumentos:
        x: Arreglo de numpy que pertenece a un conjunto de imágenes.
        y: Arreglo de numpy que pertenece a un conjunto de imágenes.
            (normalmente los mapas de segmentación de x); es opcional.
        size: Tupla de dos elementos que contiene el tamaño de las imágenes resultantes.
        num_partes: Entero que determina en cuántas partes se van a separar las imágenes.

    Regresa:
        x_imgs, y_imgs: Arreglos de numpy del conjunto de imágenes separadas en partes.
    """
    x_patches = image.PatchExtractor(patch_size=size, max_patches=num_partes, random_state=0)
    x_imgs = x_patches.transform(x)
    
    # Reajustar tamaño
    if rgb:
        nuevo_tam = list(x_imgs.shape)
    else:
        nuevo_tam = list(x_imgs.shape[:-1]) + [1]
    
    x_imgs = x_imgs.reshape(tuple(nuevo_tam))

    if y is not None:
        nuevo_tam = list(y_imgs.shape[:-1]) + [1]
        y_patches = image.PatchExtractor(patch_size=size, max_patches=num_partes, random_state=0)
        y_imgs = y_patches.transform(y)
        y_imgs = y_imgs.reshape(tuple(nuevo_tam))

        return x_imgs, y_imgs

    return x_imgs

def aumentar_imagenes_mascaras(x, y, batch_size=4, transformaciones=None, seed=6):
    """
    Toma dos conjuntos de imágenes y las transforma utilizando ImageDataGenerator de Keras.

    Argumentos:
        x: Arreglo de numpy que pertenece a un conjunto de imágenes.
        y: Arreglo de numpy que pertenece a un conjunto de imágenes.
        batch_size: Entero que determina el número de imágenes a transformar del total.
        transformaciones: Diccionario con las transformaciones a realizar sobre las imágenes.
        seed: Entero para fijar el generador de números aleatorios y hacerlo reproducible.

    Regresa:
        generador: Un generador de Python con los conjuntos de imágenes transformados.
    """
    if transformaciones is None:
        transformaciones = dict(
        rotation_range=10.0,
        height_shift_range=0.02,
        shear_range=5,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="constant"
    )

    datagen_x = ImageDataGenerator(**transformaciones)
    datagen_x.fit(x, augment=True, seed=seed)
    datagen_y = ImageDataGenerator(**transformaciones)
    datagen_y.fit(y, augment=True, seed=seed)

    x_aumentado = datagen_x.flow(x, batch_size=batch_size, seed=seed)
    y_aumentado = datagen_y.flow(y, batch_size=batch_size, seed=seed)

    generador = zip(x_aumentado, y_aumentado)

    return generador
