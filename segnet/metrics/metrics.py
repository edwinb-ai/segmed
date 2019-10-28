import tensorflow as tf


def jaccard_index(y_true, y_pred):
    """
    Jaccard index that evaluates segmentation maps and its effectiveness. If the value is one,
    the segmentation map predicted is exact.

    Args:
        y_true: TensorFlow Tensor with the ground truth.
        y_pred: TensorFlow Tensor with the predicted value.

    Returns:
        result: Scalar that determines the intersection over union value.
    """
    y_true_f = tf.reshape(y_true, shape=[-1])
    y_pred_f = tf.reshape(y_pred, shape=[-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    result = (intersection + 1.0) / (union - intersection + 1.0)
    result = tf.reduce_mean(result)

    return result


def dice_coef(y_true, y_pred, smooth=1.0):
    """
    The Sorensen-Dice coefficient is a close relative of the Jaccard index, and it should
    almost always be logged simultaneously.

    Args:
        y_true: TensorFlow Tensor with the ground truth.
        y_pred: TensorFlow Tensor with the predicted value.

    Returns:
        result: Scalar that determines the segmentation error.
    """

    y_true_f = tf.reshape(y_true, shape=[-1])
    y_pred_f = tf.reshape(y_pred, shape=[-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    numerator = (2.0 * intersection) + smooth
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    result = numerator / denom

    return result


def _static_binarization(x):
    """Take a TensorFlow Tensor object and statically binarize it, making
    the assumption that it has values between 0 and 1. If the value is >= 0.5,
    then an integer 1 is assigned, otherwise, an integer 0 is assigned.

    Args:
        x (tf.Tensor): Tensor to binarize

    Returns:
        new_x (tf.Tensor): Binarized tensor
    """
    new_x = tf.where(x >= 0.5, tf.constant([1]), tf.constant([0]))

    return new_x


def _up_dp_qp(x, y):
    """Obtain under, over and overall error segmentation rates.
    By binarizing a ground truth and predicted segmentation maps,
    this function obtains:
    - Qp, the number of pixels that should
        be included in the segmentation result but are not
    - Up, the number of pixels that should be
        excluded from the segmentation result but actually included
    - Dp, the number of pixels that should be included in the segmentation result
        and are also actually included.
    This implementation leverages logical operations and bitwise invertions for
    a faster computation of the values.

    Args:
        x (tf.Tensor): with the ground truth
        y (tf.Tensor): with the predicted value

    Returns:
        u_p (tf.Tensor): the calculated value for Up
        d_p (tf.Tensor): the calculated value for Dp
        q_p (tf.Tensor): the calculated value for Qp
    """
    y_t = tf.reshape(x, shape=[-1])
    y_p = tf.reshape(y, shape=[-1])
    y_t = _static_binarization(y_t)
    y_p = _static_binarization(y_p)
    d_p = tf.reduce_sum(y_t * y_p)
    q_p = tf.reduce_sum(y_t * tf.bitwise.invert(y_p))
    u_p = tf.reduce_sum(y_p * tf.bitwise.invert(y_t))

    return u_p, d_p, q_p


def o_rate(y_true, y_pred):
    """Calculate the over segmentation error, and cast the result
    to the input type for stability. It is defined as
    OR = Qp / (Up + Dp)

    Args:
        y_true (tf.Tensor): with the ground truth.
        y_pred (tf.Tensor): with the predicted value.

    Returns:
        result (tf.Tensor): Constant Tensor with the OR value.
    """
    u_p, d_p, q_p = _up_dp_qp(y_true, y_pred)
    result = q_p / (u_p + d_p)
    result = tf.cast(result, y_true.dtype)

    return result


def u_rate(y_true, y_pred):
    """Calculate the under segmentation error, and cast the result
    to the input type for stability. It is defined as
    UR = Up / (Up + Dp)

    Args:
        y_true (tf.Tensor): with the ground truth.
        y_pred (tf.Tensor): with the predicted value.

    Returns:
        result (tf.Tensor): Constant Tensor with the UR value.
    """
    u_p, d_p, _ = _up_dp_qp(y_true, y_pred)
    result = u_p / (u_p + d_p)
    result = tf.cast(result, y_true.dtype)

    return result


def err_rate(y_true, y_pred):
    """Calculate the overall segmentation error, and cast the result
    to the input type for stability. It is defined as
    ER = (Qp + Up) / Dp

    Args:
        y_true (tf.Tensor): with the ground truth.
        y_pred (tf.Tensor): with the predicted value.

    Returns:
        result (tf.Tensor): Constant Tensor with the ER value.
    """
    u_p, d_p, q_p = _up_dp_qp(y_true, y_pred)
    result = (q_p + u_p) / d_p
    result = tf.cast(result, y_true.dtype)

    return result

