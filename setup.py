from setuptools import setup
from setuptools import find_packages

setup(
    name="segnet",
    version="0.2",
    author="Edwin Bedolla",
    author_email="developeredwin@gmail.com",
    packages=find_packages(),
    license="LICENSE",
    url="https://github.com/DCI-NET/segnet",
    description="Applying Deep Learning to medical image segmentation tasks.",
    long_description=open("README.md").read(),
)
