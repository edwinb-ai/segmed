import preprocesamiento as prep
import metricas as mets
import modelo
import matplotlib.pyplot as plt
from skimage import io


# Cargar el modelo entrenado
unet_model = modelo.unet(input_size=(512, 512, 1))
unet_model.load_weights("unet.h5")
# Cargar los datos de prueba
X_test = prep.datos_prueba("dataset/test_images/test-volume.tif")
resultados = unet_model.predict(X_test, batch_size=1, verbose=1)
print(resultados.shape)
# for i in resultados:
plt.figure()
plt.imshow(resultados[0, :, :, 0], cmap="gray")
plt.figure()
io.imshow(X_test[0, :, :, 0])
io.show()
