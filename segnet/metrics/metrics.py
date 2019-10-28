import tensorflow as tf


def jaccard_index(y_true, y_pred):
    """
    Jaccard index that evaluates segmentation maps and its effectiveness. If the value is one,
    the segmentation map predicted is exact.

    Arguments:
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

    Arguments:
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
    new_x = tf.where(x >= 0.5, tf.constant([1]), tf.constant([0]))

    return new_x


def _up_dp_qp(x, y):
    y_t = tf.reshape(x, shape=[-1])
    y_p = tf.reshape(y, shape=[-1])
    y_t = _static_binarization(y_t)
    y_p = _static_binarization(y_p)
    d_p = tf.reduce_sum(y_t * y_p)
    q_p = tf.reduce_sum(y_t * tf.bitwise.invert(y_p))
    u_p = tf.reduce_sum(y_p * tf.bitwise.invert(y_t))

    return u_p, d_p, q_p


def o_rate(y_true, y_pred):
    u_p, d_p, q_p = _up_dp_qp(y_true, y_pred)
    result = q_p / (u_p + d_p)

    return tf.cast(result, y_true.dtype)


def u_rate(y_true, y_pred):
    u_p, d_p, _ = _up_dp_qp(y_true, y_pred)
    result = u_p / (u_p + d_p)

    return tf.cast(result, y_true.dtype)


def err_rate(y_true, y_pred):
    u_p, d_p, q_p = _up_dp_qp(y_true, y_pred)
    result = (q_p + u_p) / d_p

    return tf.cast(result, y_true.dtype)

