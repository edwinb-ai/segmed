from skimage import io
from sklearn.feature_extraction import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def extract_data(train_path, label_path=None, rgb=False):
    """Load images to memory from directories.

    Extracts data from train_path and label_path and returns a normalized
    copy of the information. These are expected to be images, and the normalization
    is just converting pixel values to a 0-1 range.

    Args:
        train_path (str): String with the path for the full images.
        label_path (str): String with the path for the segmentation maps.
        rgb (bool): Boolean value to determine if the images are color or grayscale images.

    Returns:
        X_train, y_train (ndarray): Arrays of the images and their segmentation maps, normalized.
        X_train (ndarray): Array of the images and their segmentation maps, normalized; when there
            is no `label_path`.
    """

    # Import images as a collection
    X_train = io.ImageCollection(train_path).concatenate()
    
    # Reshape the array in case it is needed
    if rgb:
        # First check if for whatever reason the array is not RGB already
        if not X_train.shape[-1] == 3:
            X_train = X_train[:, :, :, 3]
    else:
        # If not RGB, just reshape to having just one channel, grayscale
        X_train = X_train[:, :, :, np.newaxis]
    
    # Always convert to a valid type and normalize
    X_train = X_train.astype("float32")
    X_train /= 255.0

    # Do the same to the segmentation maps, if passed
    if not label_path is None:
        y_train = io.ImageCollection(label_path).concatenate()
        # The segmentation maps should always be shape (:, :, :, 1)
        y_train = y_train[:, :, :, np.newaxis]
        # Convert to a valid type and normalize
        y_train = y_train.astype("float32")
        y_train /= 255.0

        return X_train, y_train

    return X_train


def split_images(x, y=None, size=(128, 128), num_part=4):
    """Split images into smaller images.

    Takes two arrays of images, x,y, and splits them into num_part number
    of random patches.

    Args:
        x: Numpy ndarray with images.
        y: Numpy ndarray with images.
        size: Tuple with two integer values, (height, width) of the resulting patches.
        num_part: Integer value; the number of resulting patches.

    Returns:
        x_imgs, y_imgs: Numpy ndarrays with the patches of the original images.
    """
    x_patches = image.PatchExtractor(patch_size=size, max_patches=num_part, random_state=0)
    x_imgs = x_patches.transform(x)
    # Check if number of channels is the same for grayscale
    if x.shape[-1] != x_imgs.shape[-1]:
        x_imgs = x_imgs[:, :, :, np.newaxis]

    if not y is None:
        y_patches = image.PatchExtractor(patch_size=size, max_patches=num_part, random_state=0)
        y_imgs = y_patches.transform(y)

        # Check if number of channels is the same for grayscale
        if y.shape[-1] != y_imgs.shape[-1]:
            y_imgs = y_imgs[:, :, :, np.newaxis]

        return x_imgs, y_imgs

    return x_imgs

def image_mask_augmentation(x, y, batch_size=4, transformations=None, seed=6):
    """Use Keras to transform images in several ways.

    Takes two sets of images and transforms it using ImageDataGenerator from Keras.
    For more information about the possible transformation, visit the official
    Keras documentation.

    Args:
        x: Numpy ndarray with images.
        y: Numpy ndarray with images.
        batch_size: Integer value for the number of images to take simultaneously.
        transformations: Dictionary with the specified transformations to perform, using Keras API.
        seed: Integer value left fixed for reproducibility purposes.

    Returns:
        generator: A Python generator that yields both datasets transformed.
    """
    # Always perform some basic transformations
    if transformations is None:
        transformations = dict(
        rotation_range=10.0,
        height_shift_range=0.02,
        shear_range=5,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="constant"
    )

    datagen_x = ImageDataGenerator(**transformations)
    datagen_x.fit(x, augment=True, seed=seed)
    datagen_y = ImageDataGenerator(**transformations)
    datagen_y.fit(y, augment=True, seed=seed)

    x_aug = datagen_x.flow(x, batch_size=batch_size, seed=seed)
    y_aug = datagen_y.flow(y, batch_size=batch_size, seed=seed)

    generator = zip(x_aug, y_aug)

    return generator
