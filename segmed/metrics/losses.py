import tensorflow as tf
from .metrics import jaccard_index

def ternaus_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Evaluate the Ternaus loss.

    Loss inspired by TernausNet
    https://arxiv.org/abs/1801.05746
    A (hopefully) smooth and differentiable combination between binary cross-entropy
    and the Jaccard index for better segmentation training.

    Arguments:
        y_true: The ground truth.
        y_pred: The predicted value.

    Returns:
        loss: The segmentation error.
    """
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) - tf.math.log(
        jaccard_index(y_true, y_pred)
    )

    return loss