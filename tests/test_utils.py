# import pytest
from utils import utils


class TestUtilsExtractData:
    def test_correct_shape(self):
        """Test that the extracted arrays have the correct
        shape.
        """
        x, y = utils.extract_data(
            "tests/example_dataset/images_prepped_train/*.png",
            "tests/example_dataset/annotations_prepped_train/*.png",
            rgb=True,
        )
        # There are five images, RGB and grayscale for the segmentation maps
        expected_shape_x = (5, 360, 480, 3)
        expected_shape_y = (5, 360, 480, 1)

        assert x.shape == expected_shape_x
        assert y.shape == expected_shape_y

    def test_rgb(self):
        """Test that the extracted data have the correct color
        channels. Very important!!!!
        """
        # This is the same image in grayscale
        x, y = utils.extract_data(
            "tests/example_dataset/1.png",
            "tests/example_dataset/1.png",
            rgb=False,
        )
        expected_shape_x = (1, 512, 512, 1)
        expected_shape_y = (1, 512, 512, 1)

        assert x.shape == expected_shape_x
        assert y.shape == expected_shape_y

    def test_provide_label_path(self):

        x = utils.extract_data(
            "tests/example_dataset/1.png",
            rgb=False,
        )

        expected_shape_x = (1, 512, 512, 1)

        assert x.shape == expected_shape_x
        