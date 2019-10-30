from setuptools import setup
from setuptools import find_packages
import os


here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except:
    REQUIRED = []

setup(
    name="segnet",
    version="0.6",
    author="Edwin Bedolla",
    author_email="developeredwin@gmail.com",
    packages=find_packages(),
    install_requires=REQUIRED,
    license="LICENSE",
    url="https://github.com/DCI-NET/segnet",
    description="Applying Deep Learning to medical image segmentation tasks.",
    long_description=open("README.md").read(),
)
