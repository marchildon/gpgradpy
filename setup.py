#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 08:35:53 2021

@author: andremarchildon
"""

from setuptools import setup, find_packages

VERSION = '1.0.1'
DESCRIPTION = 'Gradient-enhanced Gaussian process'
LONG_DESCRIPTION = 'A python library that provides gradient-free and gradient-enhanced Gaussian processes.'

# Setting up
setup(
       # the name must match the folder name 'gpgradpy'
        name="gpgradpy",
        version=VERSION,
        author="Andre Marchildon",
        author_email="<marchildon.andre@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        #install_requires=[numpy, scipy, matplotlib, smt, tabulate]

        keywords=['python', 'Bayesian optimization', 'Gaussian process'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
        ]
)