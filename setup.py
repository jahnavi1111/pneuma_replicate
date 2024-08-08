from setuptools import find_packages, setup


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


def read_file(file):
    with open(file) as f:
        return f.read()


LONG_DESCRIPTION = read_file("README.md")
VERSION = read_file("./VERSION")
requirements = read_requirements("pneuma/requirements.txt")

setup(
    name="pneuma",
    version=VERSION,
    author="University of Chicago",
    author_email="davidalexander@uchicago.edu",
    url="https://github.com/TheDataStation/Pneuma",
    description="Pneuma is a Python package for indexing and querying tables.",
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    license="TODO License",
    packages=find_packages(include=["pneuma"]),
    # install_requires=requirements,
)
