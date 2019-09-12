# import pytest
from utils import utils


class TestUtilsExtractData:
    def test_extract_data_correct_shape(self):

        x, y = utils.extract_data(
            "tests/example_dataset/images_prepped_train/*.png",
            "tests/example_dataset/annotations_prepped_train/*.png",
        )

        expected_shape_x = (5, 360, 480, 3)
        expected_shape_y = (5, 360, 480)

        assert x.shape == expected_shape_x
        assert y.shape == expected_shape_y

