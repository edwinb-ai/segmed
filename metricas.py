from keras import backend as K
from keras.losses import binary_crossentropy


def indice_jaccard(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (
        K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0
    )


def ternaus_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) - K.log(indice_jaccard(y_true, y_pred))

    return loss


def loss_jaccard(y_true, y_pred):

    return -indice_jaccard(y_true, y_pred)
