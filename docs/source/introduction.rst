Introduction
============

This is ``segmed``, a Python package to make semantic segmentation
on medical images easier.
It uses deep learning techniques powered by ``tensorflow`` to provide neural
network architectures proven to be successful at image segmentation tasks.

Installation
------------

``segmed`` is published in the PyPI so you can install like so\: ::

    pip install segmed

The need for ``segmed``
------------------------

Image segmentation has always been a hard task for image processing,
always having to depend on complex algorithms and procedures to
obtain some fast results for benchmarking and testing whether segmentation
is the correct way to approach a given problem.

In this day and age, deep learning is a tool that is almost everywhere,
and image segmentation is not the exception.

In the medical image analysis side there has always been a need for
robust, easy-to-use, segmentation tools to achieve a benchmark or
to obtain some quick and dirty results.

``segmed`` was created with these needs in mind. A powerful, simple API that
can help scientists obtain segmentation masks from their datasets quickly
without the need to code from the scratch every possible architecture
there is.

``segmed`` strives on simplicity and expressability, so that users can get
results fast without the hassle of having to rewrite the code. We provide
simple ways to automatically reimplement current architectures with just
a few simple Python structures, and no need to learn a full deep learning
library if there's no time for it.

To learn how to use ``segmed`` continue on to the :ref:`user-guide`.
