import pytest
import tensorflow as tf
from segnet.metrics import metrics, losses
import skimage.io as skio
import numpy as np


class CreateSegmentations:
    @staticmethod
    def _create_segmentation_map():
        """Take a segmentation map, normalize it and cast it to float32.
        Then turn it to a TensorFlow tensor.
        """
        segmap = skio.imread("tests/example_dataset/1.png", as_gray=True)
        segmap = np.array(segmap / 255.0).astype(np.float32)
        segmap = tf.convert_to_tensor(segmap, dtype=tf.float32)

        return segmap

    def _create_ground_predicted(self):
        """Create a segmentation map as ground truth and modify it
        minimally so that it is considered a predicted truth.
        """
        ground_truth = self._create_segmentation_map()
        pred_truth = skio.imread("tests/example_dataset/1.png", as_gray=True)
        pred_truth = np.array(pred_truth / 255.0).astype(np.float32)
        # Change only to pixels in the image
        pred_truth[30, 40] = 0.0
        pred_truth[50, 60] = 0.0
        pred_truth = tf.convert_to_tensor(pred_truth, dtype=tf.float32)

        return ground_truth, pred_truth


class TestMetrics(CreateSegmentations):
    def test_jaccard_index(self):
        """Test that the Jaccard index evaluates to 1.0
        for the same segmentation map.
        """
        # Extract a segmentation map in grayscale and normalize
        segmap = self._create_segmentation_map()
        result = metrics.jaccard_index(segmap, segmap).numpy()

        assert pytest.approx(result == [1.0])

    def test_dice_coef(self):
        """Test that the Sorensen-Dice coefficient evaluates to 1.0
        for the same segmentation map.
        """
        # Extract a segmentation map in grayscale and normalize
        segmap = self._create_segmentation_map()

        result = metrics.dice_coef(segmap, segmap).numpy()

        assert pytest.approx(result == [1.0])

    def test_o_rate(self):
        """Test the over segmentation rate. In this case it should
        be close to one as the ground and predicted truths are almost
        equal.
        """
        ground_truth, pred_truth = self._create_ground_predicted()

        result = metrics.o_rate(ground_truth, pred_truth).numpy()

        assert pytest.approx(result == [0.0])

    def test_u_rate(self):
        """Test the under segmentation rate. In this case it should
        be close to one as the ground and predicted truths are almost
        equal.
        """
        ground_truth, pred_truth = self._create_ground_predicted()

        result = metrics.u_rate(ground_truth, pred_truth).numpy()

        assert pytest.approx(result == [0.0])

    def test_err_rate(self):
        """Test the overall segmentation rate. In this case it should
        be close to one as the ground and predicted truths are almost
        equal.
        """
        ground_truth, pred_truth = self._create_ground_predicted()

        result = metrics.err_rate(ground_truth, pred_truth).numpy()

        assert pytest.approx(result == [0.0])


class TestLosses(CreateSegmentations):
    def test_ternaus_loss(self):
        """Test that the Ternaus Loss evaluates to zero. Keep in mind
        that the result is an array, so we need to evaluate in a closed form.
        """
        segmap = self._create_segmentation_map()

        result = losses.ternaus_loss(segmap, segmap).numpy()

        assert np.all(result == 0.0)
