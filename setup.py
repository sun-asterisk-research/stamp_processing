# -*- coding: utf-8 -*-
"""
Created on  
@author: Sun* AI Research Team
"""


import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "gdown==3.12.2",
    "opencv_python==4.1.2.30",
    "torchvision==0.9.1",
    "torch==1.8.1",
    "fastai==2.3.1",
    "numpy==1.18.5",
    "PyYAML==5.4.1",
]

DISTNAME = "stamp_processing"
VERSION = "0.0.1"
LICENSE = ""
DESCRIPTION = "Stamp detector"
AUTHOR = "Sun* AI Research Team"
EMAIL = "bui.hai.minh.hieu@sun-asterisk.com"
URL = "https://github.com/sun-asterisk-research/stamp_processing/"
DOWNLOAD_URL = "https://github.com/sun-asterisk-research/stamp_processing/"


setuptools.setup(
    name=DISTNAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={
        "Bug Tracker": DOWNLOAD_URL,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
)
