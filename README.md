# GpGradPy
 Gradient-enhanced Gaussian process

This python package allows a user to build a gradient-free or a gradient-enhanced Gaussian process (GP). For an introduction to GPs the user can reference "Gaussian processes for machine learning" at https://gaussianprocess.org/gpml/chapters/RW.pdf.

## Ill-conditioning problem for gradient-enhanced GPs

Gradient-enhanced GPs have covariance matrices that suffer from ill-conditioning. Two methods are implemented that help overcome this ill-conditioning problem. The details for these methods can be found in the following papers:

 -"A Non-intrusive Solution to the Ill-Conditioning Problem of the Gradient-Enhanced Gaussian Covariance Matrix for Gaussian Processes" in the Journal of Scientific Computing that is available here: https://link.springer.com/article/10.1007/s10915-023-02190-w 
 -"A Solution to the Ill-Conditioning of Gradient- Enhanced Covariance Matrices for Gaussian Processes" on arXiv, which is accessible here: https://arxiv.org/abs/2307.05855 

To install the GpGradPy package cd to this directory and use the command "pip install ./"
You can test out the package by running the files in the directory gpgradpy/plt. The results in the paper "A Solution to the Ill-Conditioning of Gradient- Enhanced Covariance Matrices for Gaussian Processes" were run with Python 3.9.

## Abstract of latest paper

Gaussian processes provide probabilistic surrogates for various applications including classification, uncertainty quantification, and optimization. Using a gradient-enhanced covariance matrix can be beneficial since it provides a more accurate surrogate relative to its gradient-free counterpart. An acute problem for Gaussian processes, particularly those that use gradients, is the ill-conditioning of their covariance matrices. Several methods have been developed to address this problem for gradient-enhanced Gaussian processes but they have various drawbacks such as limiting the data that can be used, imposing a minimum distance between evaluation points in the parameter space, or constraining the hyperparameters. In this paper a new method is presented that applies a diagonal preconditioner to the covariance matrix along with a modest nugget to ensure that the condition number of the covariance matrix is bounded, while avoiding the drawbacks listed above. Optimization results for a gradient-enhanced Bayesian optimizer with the Gaussian kernel are compared with the use of the new method, a baseline method that constrains the hyperparameters, and a rescaling method that increases the distance between evaluation points. The Bayesian optimizer with the new method converges the optimality, i.e. the l2 norm of the gradient, an additional 5 to 9 orders of magnitude relative to when the baseline method is used and it does so in fewer iterations than with the rescaling method. The new method is available in the open source python library GpGradPy, which can be found at https://github.com/marchildon/gpgradpy/tree/paper_precon. All of the figures in this paper can be reproduced with this library.


