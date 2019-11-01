from segmed.metrics import metrics as mts
import tensorflow as tf


def train_unet(
    segmodel,
    img_path,
    mask_path,
    batch_size=16,
    epochs=25,
    steps_per_epoch=3125,
    val_split=0.2,
    optimizer=tf.keras.optimizers.Adam(),
    monitor="val_jaccard_index",
    model_file="unet_simple.h5",
    seed=1,
    show=False,
):

    """ A simple utility function for training the U-Net.
    
    Takes two paths for images and segmentation maps and rescales them, splits
    them into training and validation, while saving the best model trained.
    The paths must comply to the Keras API convention as specified in their
    documentation.

    Args:
        model (tf.keras.Model): A keras model instance.
        img_path (str): The relative path were the images are located.
        mask_path (str): The relative path were the maps are located.
        batch_size (int): Size of the batch to be processed.
        epochs (int): Number of epochs to train the model.
        steps_per_epoch (int): Total number of steps (batches of samples) to yield 
            before declaring one epoch finished and starting the next epoch.
        val_split (float): Value between 0.0 and 1.0 representing the size in percentage
            to split the dataset.
        optimizer (`Optimizer`): A Keras optimizer instance with a valid syntax from TensorFlow.
        monitor (str): Quantity to monitor during training; follow the Keras convention.
        model_file (str): File to create for the ModelCheckpoint callback from Keras.
        seed (int): Value to seed the ImageDataGenerator and always retrieve the same batch
            of pairs of images.
        show (bool): If true, plots a pair of size `batch_size` to see if the image and its
            segmentation maps are consistent. For debugging purposes only.
    Returns:
        history (`History`): A `History` object. Its `History.history` attribute is a record of
            training loss values and metrics values at successive epochs, as well as validation
            loss values and validation metrics values (if applicable).
    """


    # Rescale and convert to float32 both subsets
    data_gen_args = {
        "rescale": 1.0 / 255.0,
        "validation_split": val_split,
        "dtype": tf.float32,
    }

    # Crea the training generators with the defined transformations
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    # Take images from directories
    image_generator_train = image_datagen.flow_from_directory(
        img_path,
        class_mode=None,
        color_mode="rgb",
        batch_size=batch_size,
        seed=seed,
        subset="training",
    )
    mask_generator_train = mask_datagen.flow_from_directory(
        mask_path,
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        seed=seed,
        subset="training",
    )
    # Combine both generators
    # This need to be a generator of generators, see the following
    # https://github.com/tensorflow/tensorflow/issues/32357
    train_generator = (
        pair for pair in zip(image_generator_train, mask_generator_train)
    )

    # And now the validation generators
    image_generator_val = image_datagen.flow_from_directory(
        img_path,
        class_mode=None,
        color_mode="rgb",
        batch_size=batch_size,
        seed=seed,
        subset="validation",
    )
    mask_generator_val = mask_datagen.flow_from_directory(
        mask_path,
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        seed=seed,
        subset="validation",
    )
    # Combine both generators, with same issue as before
    val_generator = (pair for pair in zip(image_generator_val, mask_generator_val))

    # DEBUGGING ONLY, for checking that both image and maps are batched together
    if show:
        import matplotlib.pyplot as plt

        for i, j in train_generator:
            plt.figure(0)
            plt.imshow(i[0, ...])
            plt.figure(1)
            plt.imshow(j[0, ..., 0])
            plt.show()

    # Define the input size in the model and build the model
    segmodel.input_size = next(image_generator_train)[0].shape
    model = segmodel.collect()
    # Define the checkpoint callback, always maximum mode for custom metrics
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_file, monitor=monitor, verbose=1, save_best_only=True, mode="max"
    )

    # Compile the model with an and custom metrics
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=optimizer,
        metrics=[mts.jaccard_index, mts.dice_coef],
    )

    # Create history of model and return it
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[checkpoint],
        verbose=1,
        validation_data=val_generator,
        validation_steps=steps_per_epoch,
    )

    return history
