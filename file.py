import pytest
import tensorflow as tf
from metrics import metrics
import skimage.io as skio
import os


def jaccard_index():

    segmentation_map = skio.imread(
            "tests/example_dataset/annotations_prepped_train/0001TP_006690.png"
    )
    print(segmentation_map)
    segmentation_map = tf.convert_to_tensor(segmentation_map, dtype=tf.float32)

    result = metrics.jaccard_index(segmentation_map, segmentation_map)
    print(result)
    # assert pytest.approx(result) == tf.constant([0.0], dtype=tf.float32)


if __name__ == "__main__":
    jaccard_index()

