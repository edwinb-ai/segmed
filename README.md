# SegNet [![Build Status](https://travis-ci.org/DCI-NET/segnet.svg?branch=master)](https://travis-ci.org/DCI-NET/segnet)

This is a collection of Deep Learning semantic segmentation models to use for
specific tasks, namely medical images, cells, histological data and related.

## Models

The models here presented are just two, namely the [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
and the [MultiResUNet](https://arxiv.org/pdf/1902.04049.pdf). These models have been a very good
application of Fully Convolutional Networks to the image segmentation task, and are very
well suited for it.

## Implementation

Everything is implemented with [TensorFlow 2.0](tensorflow.org), using the newly acquired Keras API
within TensorFlow. This allows for the flexibility and completeness of using the full TensorFlow
library while still having very good scripting capabilities with Keras.

## Dependencies

There are several ways to install this package.

### Using `pip`

The easiest way to install this package is using `pip` with the following command

    pip install git+https://github.com/DCI-NET/segnet

although it is _highly encouraged_ to do this inside a virtual environment.

If using [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
then this is the **preferred** way to use the package.

**IMPORTANT**: When using Colaboratory one must always install `TensorFlow 2.0` first and then install `segnet`,
i.e. this package, using the following commands in a cell within a Colaboratory notebook:
```
!pip install --upgrade pip
!pip install tensorflow
!pip install git+https://github.com/DCI-NET/segnet
```

### Using `pipenv`

[`pipenv`](https://pipenv.readthedocs.io/en/latest/) is the **preferred** way to install this package on a local machine. One just needs
to run the following:
```
# create a new directory for this
mkdir segnet_test
# install and create a new environment
pipenv install git+https://github.com/DCI-NET/segnet#egg=segnet
# activate the newly created shell with everything installed
pipenv shell
```

`pipenv` facilitates the use of dependencies and virtual environments, making life easier.

### Using `conda`

This bundle also comes with a
[conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for those that use the [Anaconda distribution](https://www.anaconda.com/distribution/).

To install the environment make sure you have [conda](https://conda.io/en/latest/) installed, then run the following

    conda env create -f segnet_env.yml

this should ask you to confirm the installation, say yes and proceed with the installation. After that, activate the newly
created environment

    conda activate segnet

and now you are ready to run the code within this repository.

## Unit tests

This repository has some unit tests available that should be running constantly in the background,
and the status of the current code coverage is displayed in the badge above (the one right to the title).

One can manually run the tests, too. Having [pytest](https://pytest.org/en/latest/) (one of the dependencies) 
installed one just needs to run

    pytest

and the test should start running, with a few import and API warnings, but everything should pass.

## Examples

This repository also has some (_very_) barebones examples of how to use and employ these models.
However, most were run in a private Colaboratory account and most of the data cannot be used.
These examples should only be used to have a basic idea of how to run and create an image segmentation
pipeline, but you will _not_ be able to rerun the notebooks.

The reason for this is that most of the datasets are **very** large, so they cannot be bundled
with this repository; some other datasets **cannot** be redistributed as per request of the original authors.

### Demo

For completeness, here is a simple example. Assuming you have followed the instructions and everything is installed
correctly, you can do the following to train a simple U-Net model:
```python
from segnet.train import train_segnet
from segnet.models import Unet

# Define some example hyperparameters
batch_size = 8
epochs = 50
steps_per_epoch=100

# Declare the paths to use (following the Keras convention)
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#fit_generator
data_path = "general/path/to/images"
img_path = data_path + "augmented/images/path"
masks_path = data_path + "augmented/masks/path"
model_file = "path/to/save/model/unet_model.h5"

# Create a Unet (custom) model with a regularizer and
# a different value for the dropout
custom_params = {
    "activation": "relu",
    "padding": "same",
    "dropout": 0.6,
    "l2_reg": 0.995
}
model = Unet(variant="custom", parameters=custom_params)
# Train the model!
history = train_segnet(
    model,
    img_path,
    masks_path,
    batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    model_file=model_file,
)
```
Now, the training should have started and you're good to go!

## Datasets

The datasets employed in some of the parts of this repository are the following:

- [ISBI 2012 Challenge](http://brainiac2.mit.edu/isbi_challenge/home)
- [Motion-based Segmentation and Recognition Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
- [DRIVE: Digital Retinal Images for Vessel Extraction](https://www.isi.uu.nl/Research/Databases/DRIVE/)
- [Fast and Robust Segmentation of White Blood Cell Images by Self-supervised Learning](https://github.com/zxaoyou/segmentation_WBC)