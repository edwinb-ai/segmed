from keras.models import load_model
import preprocesamiento as prep


# Cargar el modelo entrenado
unet_model = load_model("unet.h5")
# Cargar los datos de prueba
X_test = prep.datos_prueba("dataset/train_images/test-volume.tif")
resultados = unet_model.predict(X_test)
print(resultados.shape)
