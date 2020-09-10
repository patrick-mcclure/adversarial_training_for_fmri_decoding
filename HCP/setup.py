#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup file for taskandyeshallreceive."""

import os

from setuptools import find_packages, setup

NAME = 'taskandyeshallperceive'
DESCRIPTION = 'A bayesian network to assess how brainy a scan is.'
URL = 'https://github.com/mcclureps/taskandyeshallreceive'
EMAIL = 'patrick.mcclure@nih.gov'
AUTHOR = 'taskandyeshallreceive developers'

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with open(os.path.join(here, 'README.md'), encoding='utf-8') as fp:
    long_description = fp.read()

with open(os.path.join(here, 'requirements.txt')) as fp:
    REQUIRED = fp.readlines()

about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require={
        'cpu': ["tensorflow==1.8.0"],
        'gpu': ["tensorflow-gpu==1.8.0"],
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    entry_points={
        "console_scripts": [
            "taskandyeshallperceive=taskandyeshallperceive.cli:main"
        ]
    },
)
