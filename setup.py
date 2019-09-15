from distutils.core import setup

setup(
    name="segnet",
    version="0.1",
    author="Edwin Bedolla",
    author_email="developeredwin@gmail.com",
    packages=["segnet"],
    license="LICENSE",
    description="Applying Deep Learning to medical image segmentation tasks.",
    long_description=open("README.md").read(),
)
