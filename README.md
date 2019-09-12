# SegNet [![Build Status](https://travis-ci.org/DCI-NET/segnet.svg?branch=master)](https://travis-ci.org/DCI-NET/segnet)

This is a collection of Deep Learning semantic segmentation models to use for
specific tasks, namely medical images, cells, histological data and related.

## Models

The models here presented are just two, namely the [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
and the [MultiResUNet](https://arxiv.org/pdf/1902.04049.pdf). These models have been a very good
application of Fully Convolutional Networks to the image segmentation task, and are very
well suited for the task.

## Implementation

Everything is implemented with [TensorFlow 2.0](tensorflow.org), using the newly acquired Keras API
within TensorFlow. This allows for the flexibility and completeness of using the full TensorFlow
library will still having very good scripting capabilities with Keras.

## Executing the code

### Dependencies

This bundle comes with a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
file that **should** be installed in a local machine. If using [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
then there is almost _no_ need to use this.

For installing the environment make sure you have [conda](https://conda.io/en/latest/) installed, then run the following

    conda env create -f segnet_env.yml

this should ask you to confirm the installation, say yes and proceed with the installation

    conda activate segnet

and now you are ready to run the code in this repository.

### Unit tests

This repository has some unit tests available that should be running constantly in the background,
and the status of the current code coverage is displayed in the badge above (the one right to the title).

One can manually run the tests, too. Having [pytest](https://pytest.org/en/latest/) (one of the dependencies) 
installed one just needs to run

    pytest test/

and the test should start running, with a few import and API warnings, but everything should pass.

## Examples

This repository also has some (_very_) barebones examples of how to use and employ these models.
However, most were run in a private Colaboratory account and most of the data cannot be used.
These examples should only be used to have a basic idea of how to run and create an image segmentation
pipeline, but you will _not_ be able to rerun the notebooks.

The reason for this is that most of the datasets are **very** large, so they cannot be bundled
with this repository; some other datasets **cannot** be redistributed as per request of the original authors.

## Datasets

The datasets employed in some of the parts of this repository are the following:

- [ISBI 2012 Challenge](http://brainiac2.mit.edu/isbi_challenge/home)
- [Motion-based Segmentation and Recognition Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
- [DRIVE: Digital Retinal Images for Vessel Extraction](https://www.isi.uu.nl/Research/Databases/DRIVE/)
- [Fast and Robust Segmentation of White Blood Cell Images by Self-supervised Learning](https://github.com/zxaoyou/segmentation_WBC)