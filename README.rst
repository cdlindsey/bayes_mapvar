.. image:: docs/title.png
  :alt: bayes_mapvar logo
|PyPI Version|

Bayesian MAP / Variance Estimation
==================================
**bayes_mapvar** is a Python package that provides tools for Maximum A Posterior (MAP) estimation and posterior variance estimation.
TensorFlow Probability and SciPy are leveraged to allow fast and efficient estimation for Bayesian models.

Installation
------------
1. pip install bayes_mapvar 

or

1. Clone the repository.
2. Navigate to the project root directory.
3. Create the virtual environment with:

> `. bin/setup_venv.sh`

Documentation
-------------

The documentation for the latest release is at

https://cdlindsey.github.io/bayes_mapvar/mapvar.html

Usage and Examples
------------------

We predict oscar winners in this google collab `notebook <https://colab.research.google.com/drive/1t87-8UHzC0e8rGLNwN7YRoVamcdz74Ci#scrollTo=JjlHvSQZbRKB>`_.

Testing
-------
To run unit tests, run this from the project root directory:

> `. bin/run_unit_tests.sh`

.. |PyPI Version| image:: https://img.shields.io/pypi/v/bayes_mapvar.svg
   :target: https://pypi.org/project/bayes_mapvar/
