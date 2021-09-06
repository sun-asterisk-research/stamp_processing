# -*- coding: utf-8 -*-
"""
Created on  
@author: Hieu Bui - Sun* AI Research Team
"""


import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
INSTALL_REQUIRES = (
    ['numpy>=1.20.3', 'opencv-python==4.5.1.48', 'torch>=1.8.1', 'torchvision>=0.9.1',
    'pdf2image==1.14.0']
)

DISTNAME = "stamp_processing"
VERSION = "0.0.1"
LICENSE = ""
DESCRIPTION = "Stamp detector"
AUTHOR = "Hieu Bui"
EMAIL = ""
URL = ""
DOWNLOAD_URL = ""

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
    package_dir={"": "stamp_detector"},
    packages=setuptools.find_packages(where="stamp_detector"),
    python_requires=">=3.6",
    zip_safe=False,
    install_requires=INSTALL_REQUIRES
)