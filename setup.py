# from distutils.core import setup
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bootcamp",
    version="0.0.1",
    author="-",
    author_email="-",
    description="-",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['bootcamp'],
)

