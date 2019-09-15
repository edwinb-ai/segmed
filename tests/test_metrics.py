import pytest
import tensorflow as tf
from segnet.metrics import metrics
import skimage.io as skio
import numpy as np


def test_jaccard_index():
    """Test that the Jaccard index evaluates to 1.0
    for the same segmentation map.
    """
    # Extract a segmentation map in grayscale and normalize
    segmap = skio.imread("tests/example_dataset/1.png", as_gray=True)
    segmap = np.array(segmap / 255.0).astype(np.float32)
    # Use eager execution for TensorFlow
    segmap = tf.convert_to_tensor(segmap, dtype=tf.float32)
    result = metrics.jaccard_index(segmap, segmap).numpy()

    assert pytest.approx(result == [1.0])


def test_dice_coef():
    """Test that the Sorensen-Dice coefficient evaluates to 1.0
    for the same segmentation map.
    """
    # Extract a segmentation map in grayscale and normalize
    segmap = skio.imread("tests/example_dataset/1.png", as_gray=True)
    segmap = np.array(segmap / 255.0).astype(np.float32)

    segmap = tf.convert_to_tensor(segmap, dtype=tf.float32)

    result = metrics.dice_coef(segmap, segmap).numpy()

    assert pytest.approx(result == [1.0])


def test_ternaus_loss():
    """Test that the Ternaus Loss evaluates to zero. Keep in mind
    that the result is an array, so we need to evaluate in a closed form.
    """

    segmap = skio.imread("tests/example_dataset/1.png", as_gray=True)
    segmap = np.array(segmap / 255.0).astype(np.float32)

    segmap = tf.convert_to_tensor(segmap, dtype=tf.float32)

    result = metrics.ternaus_loss(segmap, segmap).numpy()

    assert np.all(result == 0.0)
