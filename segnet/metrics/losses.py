import tensorflow as tf
from .metrics import jaccard_index

def ternaus_loss(y_true, y_pred):
    """
    Loss inspired by TernausNet
    https://arxiv.org/abs/1801.05746
    
    A (hopefully) smooth and differentiable combination between binary cross-entropy
    and the Jaccard index for better segmentation training.

    Arguments:
        y_true: TensorFlow Tensor with the ground truth.
        y_pred: TensorFlow Tensor with the predicted value.

    Returns:
        loss: Scalar that determines the segmentation error.
    """
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) - tf.math.log(
        jaccard_index(y_true, y_pred)
    )

    return loss