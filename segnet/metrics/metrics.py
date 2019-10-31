import tensorflow as tf


def jaccard_index(y_true, y_pred):
    """Evaluate the Jaccard index.

    Assuming that `y_true` and `y_pred` are images,
    the Jaccard index evaluates segmentation maps and its effectiveness.
    If the value is one, the segmentation map predicted is exact.

    Args:
        y_true: The ground truth.
        y_pred: The predicted value.

    Returns:
        result: Scalar that determines the intersection over union value.
    """
    y_true_f = tf.reshape(y_true, shape=[-1])
    y_pred_f = tf.reshape(y_pred, shape=[-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    result = (intersection + 1.0) / (union - intersection + 1.0)
    result = tf.reduce_mean(result)
    result = tf.cast(result, tf.float32)

    return result


def dice_coef(y_true, y_pred, smooth=1.0):
    """Evaluate the Sorensen-Dice coefficient.

    Assuming that `y_true` and `y_pred` are images.
    The Sorensen-Dice coefficient is a close relative of the Jaccard index, and it should
    almost always be logged simultaneously.

    Args:
        y_true: The ground truth.
        y_pred: The predicted value.

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
    """Take a TensorFlow Tensor object and statically binarize it.
    
    This function makes the assumption that `x`
     has values between 0 and 1. If the element value of `x` is >= 0.5,
    then an integer 1 is assigned, otherwise, an integer 0 is assigned.

    Args:
        x: Tensor to binarize

    Returns:
        new_x: Binarized tensor
    """
    new_x = tf.where(x >= 0.5, 1, 0)
    new_x = tf.cast(new_x, tf.float32)

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
    This implementation leverages logical operations and invertions for
    a faster computation of the values.

    Args:
        x: The ground truth
        y: The predicted value

    Returns:
        u_p: the calculated value for Up
        d_p: the calculated value for Dp
        q_p: the calculated value for Qp
    """
    y_t = tf.reshape(x, shape=[-1])
    y_p = tf.reshape(y, shape=[-1])
    y_t = _static_binarization(y_t)
    y_p = _static_binarization(y_p)
    d_p = tf.reduce_sum(y_t * y_p)
    q_p = tf.reduce_sum(y_t * (1.0 - y_p))
    u_p = tf.reduce_sum(y_p * (1.0 - y_t))

    return u_p, d_p, q_p


def o_rate(y_true, y_pred):
    """Calculate the over segmentation error.
    
    Assuming `y_true` and `y_pred` are images, this function calculates
    the overall segmentation error, OR, defined as:

    Args:
        y_true: with the ground truth.
        y_pred: with the predicted value.

    Returns:
        result: Constant Tensor with the OR value.
    """
    u_p, d_p, q_p = _up_dp_qp(y_true, y_pred)
    # Add a smoothing factor of 1.0 to prevent division by zero
    result = q_p / (u_p + d_p + 1.0)
    # Use the number of images as a means of normalizing the result
    num_data = tf.cast(tf.shape(y_true), result.dtype)
    result = tf.cast(result / num_data[0], tf.float32)

    return result


def u_rate(y_true, y_pred):
    """Calculate the under segmentation error.
    
    Assuming `y_true` and `y_pred` are images, this function calculates
    the overall segmentation error, UR, defined as:

    UR = Up / (Up + Dp)

    Args:
        y_true: with the ground truth.
        y_pred: with the predicted value.

    Returns:
        result: Constant Tensor with the UR value.
    """
    u_p, d_p, _ = _up_dp_qp(y_true, y_pred)
    # Add a smoothing factor of 1.0 to prevent division by zero
    result = u_p / (u_p + d_p + 1.0)
    # Use the number of images as a means of normalizing the result
    num_data = tf.cast(tf.shape(y_true), result.dtype)
    result = tf.cast(result / num_data[0], tf.float32)

    return result


def err_rate(y_true, y_pred):
    """Calculate the overall segmentation error.
    
    Assuming `y_true` and `y_pred` are images, this function calculates
    the overall segmentation error, ER, defined as:

    ER = (Qp + Up) / Dp

    Args:
        y_true: with the ground truth.
        y_pred: with the predicted value.

    Returns:
        result: Constant Tensor with the ER value.
    """
    u_p, d_p, q_p = _up_dp_qp(y_true, y_pred)
    # Add a smoothing factor of 1.0 to prevent division by zero
    result = (q_p + u_p) / (d_p + 1.0)
    # Use the number of images as a means of normalizing the result
    num_data = tf.cast(tf.shape(y_true), result.dtype)
    result = tf.cast(result / num_data[0], tf.float32)
    result = tf.math.abs(result)

    return result

