import pytest
import tensorflow as tf
from metrics import metrics
import skimage.io as skio
import numpy as np


def test_jaccard_index():
    sess = tf.compat.v1.Session()

    segmap = np.zeros((128, 128), dtype=np.int32)
    segmap[28:71, 35:85] = 1
    segmap[10:25, 30:45] = 2
    segmap[10:25, 70:85] = 3
    segmap[10:110, 5:10] = 4
    segmap[118:123, 10:110] = 5

    segmap = tf.convert_to_tensor(segmap, dtype=tf.float32)

    result = metrics.jaccard_index(segmap, segmap)

    assert pytest.approx(sess.run(result)) == 0.0
