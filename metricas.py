from keras import backend as K
from keras.losses import binary_crossentropy


def indice_jaccard(y_true, y_pred):
    """
    Métrica de Jaccard (intersección sobre unión, IoU) para determinar
    la eficacia de la segmentación. Es una métrica personalizada para Keras.

    Argumentos:
        y_true: Arreglo de numpy o tensor de TensorFlow que contiene el resultado real.
        y_true: Arreglo de numpy o tensor de TensorFlow que contiene el resultado obtenido del modelo.

    Regresa:
        resultado: Un valor flotante proveniente de hacer la medición.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    interseccion = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    resultado = (interseccion + 1.0) / (union - interseccion + 1.0)

    resultado = K.mean(resultado)

    return resultado


def ternaus_loss(y_true, y_pred):
    """
    Pérdida inspirada por TernausNet
    https://arxiv.org/abs/1801.05746
    
    Es una combinación entre entropía cruzada binaria y el logaritmo del índice de Jaccard.

    Argumentos:
        y_true: Arreglo de numpy o tensor de TensorFlow que contiene el resultado real.
        y_true: Arreglo de numpy o tensor de TensorFlow que contiene el resultado obtenido del modelo.

    Regresa:
        loss: Un valor flotante proveniente de hacer la medición.
    """
    loss = binary_crossentropy(y_true, y_pred) - K.log(indice_jaccard(y_true, y_pred))

    return loss
