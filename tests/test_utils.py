import pytest
from segnet.utils import utils


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

    def test_grayscale(self):
        """Test that the extracted data have the correct color
        channels. Very important!!!!
        """
        # This is the same image in grayscale
        x, y = utils.extract_data(
            "tests/example_dataset/1.png", "tests/example_dataset/1.png", rgb=False
        )
        expected_shape_x = (1, 512, 512, 1)
        expected_shape_y = (1, 512, 512, 1)

        assert x.shape == expected_shape_x
        assert y.shape == expected_shape_y

    def test_provide_label_path(self):
        """Test that it can import just a single batch of images.
        This is very useful for importing just testing data, where we do
        not need segmentation maps.
        """

        x = utils.extract_data("tests/example_dataset/1.png", rgb=False)

        expected_shape_x = (1, 512, 512, 1)

        assert x.shape == expected_shape_x

    def test_normalized(self):
        """Test that the images are actually normalized within
        range 0-1. Very important!!!
        """

        # This is the same image in grayscale
        x, y = utils.extract_data(
            "tests/example_dataset/1.png", "tests/example_dataset/1.png", rgb=False
        )

        assert (x >= 0.0).all() and (x <= 1.0).all()
        assert (y >= 0.0).all() and (y <= 1.0).all()


class TestSplitData:
    def test_correct_shape_rgb(self):
        """Test that the extracted arrays have the correct
        shape when they are RGB.
        """
        # Both sets of images are RGB
        x, y = utils.extract_data(
            "tests/example_dataset/images_prepped_train/*.png",
            "tests/example_dataset/images_prepped_train/*.png",
            rgb=True,
        )
        x, y = utils.split_images(x, y, size=(60, 80), num_part=6)
        # There are five images and RGB, so there is a total of 30 patches
        expected_shape_x = (30, 60, 80, 3)
        expected_shape_y = (30, 60, 80, 3)

        assert x.shape == expected_shape_x
        assert y.shape == expected_shape_y

    def test_correct_shape_grayscale(self):
        """Test that the extracted arrays have the correct
        shape when they are RGB and grayscale.
        """
        # First set is RGB, second one is grayscale
        x, y = utils.extract_data(
            "tests/example_dataset/images_prepped_train/*.png",
            "tests/example_dataset/annotations_prepped_train/*.png",
            rgb=True,
        )
        x, y = utils.split_images(x, y, size=(60, 80), num_part=6)
        expected_shape_x = (30, 60, 80, 3)
        expected_shape_y = (30, 60, 80, 1)

        assert x.shape == expected_shape_x
        assert y.shape == expected_shape_y

    def test_single_array_rgb(self):
        """Test that it can import just a single batch of images.
        This is very useful for importing just testing data, where we do
        not need segmentation maps. RGB version.
        """
        x = utils.extract_data(
            "tests/example_dataset/images_prepped_train/*.png", rgb=True
        )
        x = utils.split_images(x, size=(60, 80), num_part=6)
        expected_shape_x = (30, 60, 80, 3)

        assert x.shape == expected_shape_x

    def test_single_array_grayscale(self):
        """Test that it can import just a single batch of images.
        This is very useful for importing just testing data, where we do
        not need segmentation maps. Grayscale version
        """
        x = utils.extract_data("tests/example_dataset/annotations_prepped_train/*.png")
        x = utils.split_images(x, size=(60, 80), num_part=6)
        expected_shape_x = (30, 60, 80, 1)

        assert x.shape == expected_shape_x


class TestAugmentations:
    def test_batch_size(self):
        """Test that the batch size defined is the actual
        value obtained.
        """
        x, y = utils.extract_data(
            "tests/example_dataset/images_prepped_train/*.png",
            "tests/example_dataset/annotations_prepped_train/*.png",
            rgb=True,
        )
        gen = utils.image_mask_augmentation(x, y, batch_size=2)
        # There are five images and RGB, so there is a total of 30 patches
        expected_shape_x = (2, 360, 480, 3)
        expected_shape_y = (2, 360, 480, 1)

        x_gen, y_gen = next(gen)

        assert x_gen.shape == expected_shape_x
        assert y_gen.shape == expected_shape_y
