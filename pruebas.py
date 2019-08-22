import preprocesamiento as prep
import matplotlib.pyplot as plt
from skimage import io


# Algunas pruebas de imágenes
imagenes_entrenamiento = "dataset/train_images/train-volume.tif"
etiquetas_entrenamiento = "dataset/train_images/train-labels.tif"
X, y, imagen_vista = prep.extraer_datos(
    imagenes_entrenamiento, etiquetas_entrenamiento, show=True
)
# Extracción de pedazos de imagen
x_patches, y_patches = prep.imagen_en_partes(
    X[imagen_vista], y[imagen_vista], size=(128, 128), num_partes=4
)
for i, j in zip(x_patches, y_patches):
    plt.figure()
    io.imshow(i)
    plt.figure()
    io.imshow(j)
io.show()
