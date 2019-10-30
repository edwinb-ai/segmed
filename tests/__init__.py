import skimage.io as skio
import numpy as np


class SimpleDataset:
    @staticmethod
    def _create_dataset():
        """Creates a very simple dataset from within this directory
        by taking some training and testing sets, normalizes them
        and crops them.
        """
        # Import some sample images from within the directory
        x_train = skio.ImageCollection(
            "tests/example_dataset/images_prepped_train/*.png"
        ).concatenate()
        y_train = skio.ImageCollection(
            "tests/example_dataset/annotations_prepped_train/*.png"
        ).concatenate()
        x_test = skio.ImageCollection(
            "tests/example_dataset/images_prepped_test/*.png"
        ).concatenate()
        y_test = skio.ImageCollection(
            "tests/example_dataset/annotations_prepped_test/*.png"
        ).concatenate()
        # Crop the images to 256x256 and convert them to float32
        x_train = x_train[:, :256, :256, :].astype(np.float32)
        y_train = y_train[:, :256, :256, None].astype(np.float32)
        x_test = x_test[:, :256, :256, :].astype(np.float32)
        y_test = y_test[:, :256, :256, None].astype(np.float32)
        # Normalize the images
        x_train /= 255.0
        y_train /= 255.0
        x_test /= 255.0
        y_test /= 255.0

        return x_train, x_test, y_train, y_test
