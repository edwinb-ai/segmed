import tensorflow as tf


def jaccard_index(y_true, y_pred):
    """
    Métrica de Jaccard (intersección sobre unión, IoU) para determinar
    la eficacia de la segmentación. Es una métrica personalizada para Keras.

    Argumentos:
        y_true: Arreglo de numpy o tensor de TensorFlow que contiene el resultado real.
        y_true: Arreglo de numpy o tensor de TensorFlow que contiene el resultado obtenido del modelo.

    Regresa:
        resultado: Un valor flotante proveniente de hacer la medición.
    """
    y_true_f = tf.reshape(y_true, shape=[-1])
    y_pred_f = tf.reshape(y_pred, shape=[-1])
    interseccion = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    resultado = (interseccion + 1.0) / (union - interseccion + 1.0)

    resultado = tf.reduce_mean(resultado)

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
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) - tf.log(
        jaccard_index(y_true, y_pred)
    )

    return loss


def dice_coef(y_true, y_pred, smooth=1.0):

    y_true_f = tf.reshape(y_true, shape=[-1])
    y_pred_f = tf.reshape(y_pred, shape=[-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    numerator = 2.0 * intersection + smooth
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth

    return numerator / denom