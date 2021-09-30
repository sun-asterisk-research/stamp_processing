# -*- coding: utf-8 -*-
"""
Created on  
@author: Sun* AI Research Team
"""
import sys

from setuptools import find_packages, setup

assert sys.version_info >= (3, 7, 0), "stamp_processing requires Python 3.7+"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "gdown",
    "Pillow<=8.2.0"
    "opencv-python>=4.1.2.30",
    "torch==1.8.1",
    "torchvision==0.9.1",
    "fastai==2.3.1",
    "numpy>=1.20",
    "PyYAML==5.4.1",
]

DISTNAME = "stamp_processing"
DESCRIPTION = "Stamp processing package"
AUTHOR = "Sun* AI Research Team"
EMAIL = "sun.converter.team@gmail.com"
URL = "https://github.com/sun-asterisk-research/stamp_processing/"
DOWNLOAD_URL = "https://github.com/sun-asterisk-research/stamp_processing/"


setup(
    name=DISTNAME,
    author=AUTHOR,
    author_email=EMAIL,
    use_scm_version={
        "write_to": "stamp_processing/__version__.py",
        "version_scheme": "guess-next-dev",
        "local_scheme": "no-local-version",
    },
    setup_requires=["setuptools_scm"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={
        "Bug Tracker": "https://github.com/sun-asterisk-research/stamp_processing/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where=".", exclude=["tests"]),
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.7",
    include_package_data=True
)
