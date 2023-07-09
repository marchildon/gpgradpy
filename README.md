# GpGradPy
 Gradient-enhanced Gaussian process

This python package allows a user to build a gradient-free or a gradient-enhanced Gaussian process (GP). Gradient-enhanced GPs have covariance matrices that suffer from ill-conditioning. Two methods are implemented that help overcome this ill-conditioning problem. The details for these methods can be found in the papers:
    "A Non-intrusive Solution to the Ill-Conditioning Problem of the Gradient-Enhanced Gaussian Covariance Matrix for Gaussian Processes" in the Journal of Scientific Computing.  
    "A Solution to the Ill-Conditioning of Gradient- Enhanced Covariance Matrices for Gaussian Processes" on arXiv

To install the GpGradPy package cd to this directory and use the command "pip install ./"
You can test out the package by running the files in the directory gpgradpy/plt