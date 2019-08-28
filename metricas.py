from keras import backend as K
from keras.losses import binary_crossentropy


def indice_jaccard(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    interseccion = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    resultado = (interseccion + 1.0) / (union - interseccion + 1.0)

    return K.mean(resultado)


def ternaus_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) - K.log(indice_jaccard(y_true, y_pred))

    return loss


def loss_jaccard(y_true, y_pred):

    return 1.0 - indice_jaccard(y_true, y_pred)
