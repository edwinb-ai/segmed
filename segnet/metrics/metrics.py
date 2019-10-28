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

def O_Rate(y_true,y_pred):
  y_t=tf.reshape(y_true,shape=[-1])
  y_p=tf.reshape(y_pred,shape=[-1])
  uno=tf.constant(1.0,dtype=tf.float32)
  y_true_b=tf.round(y_t+0.1)             #Agregamos un pequeño bias para añadir a la membrana en la segmentación
  y_pred_b=tf.round(y_p+0.1)
  Dp=tf.reduce_sum(y_true_b*y_pred_b)
  Qp=tf.reduce_sum(y_true_b*(uno-y_pred_b))
  Up=tf.reduce_sum(y_pred_b*(uno-y_true_b))
  return (Qp/(Up+Dp))

def U_Rate(y_true,y_pred):
  y_t=tf.reshape(y_true,shape=[-1])
  y_p=tf.reshape(y_pred,shape=[-1])
  uno=tf.constant(1.0,dtype=tf.float32)
  y_true_b=tf.round(y_t+0.1)                #Agregamos un pequeño bias para añadir a la membrana en la segmentación
  y_pred_b=tf.round(y_p+0.1)
  Dp=tf.reduce_sum(y_true_b*y_pred_b) 
  Qp=tf.reduce_sum(y_true_b*(uno-y_pred_b))
  Up=tf.reduce_sum(y_pred_b*(uno-y_true_b))
  return (Up/(Up+Dp))

def Err_rate(y_true,y_pred):
  y_t=tf.reshape(y_true,shape=[-1])
  y_p=tf.reshape(y_pred,shape=[-1])
  uno=tf.constant(1.0,dtype=tf.float32)
  y_true_b=tf.round(y_t+0.1)                #Agregamos un pequeño bias para añadir a la membrana en la segmentación
  y_pred_b=tf.round(y_p+0.1)
  Dp=tf.reduce_sum(y_true_b*y_pred_b)
  Qp=tf.reduce_sum(y_true_b*(uno-y_pred_b))
  Up=tf.reduce_sum(y_pred_b*(uno-y_true_b))
  return ((Qp+Up)/Dp)
