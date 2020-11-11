# wildboar
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)
[![Build Status](https://travis-ci.com/isaksamsten/wildboar.svg?branch=master)](https://travis-ci.com/isaksamsten/wildboar)
[![PyPI version](https://badge.fury.io/py/wildboar.svg)](https://badge.fury.io/py/wildboar)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4264063.svg)](https://doi.org/10.5281/zenodo.4264063)

[wildboar](https://isaksamsten.github.io/wildboar/) is a Python module for temporal machine learning and fast
distance computations built on top of
[SciKit-Learn](https://scikit-learn.org) and [Numpy](https://numpy.org)
distributed under the GNU General Public License Version 3.

It is currently maintained by Isak Samsten

## Installation

### Dependencies

wildboar requires:

 * Python (>= 3.6)
 * NumPy (>= 1.8.2)
 * SciPy (>= 0.13.3)
 
Some parts of wildboar is implemented using Cython. Hence, compilation
requires:

 * Cython (>= 0.28)

### Current version

- Current release: 1.0.3
- Current development release: 1.0.3dev

### Binaries

`wildboar` is available through `pip` and can be installed with:

    pip install wildboar

Universal binaries are compiled for GNU/Linux and Python 3.6 and 3.7.

### Compilation

If you already have a working installation of NumPy, SciPy and Cython,
compiling and installing wildboar is as simple as:

    python setup.py install
	
To install the requirements, use:

    pip install -r requirements.txt
	

## Development

Contributions are welcome. Pull requests are encouraged to be
formatted according to
[PEP8](https://www.python.org/dev/peps/pep-0008/), e.g., using
[yapf](https://github.com/google/yapf).

## Usage

    from wildboar import ShapeletForestClassifier
    from wildboar.dataset import load_two_lead_ecg
    x_train, x_test, y_train, y_test = load_two_lead_ecg(merge_train_test=False)
    c = ShapeletForestClassifier()
    c.fit(x_train, y_train)
    f.score(x_test, y_test)

## Source code

You can check the latest sources with the command:

    git clone https://github.com/isakkarlsson/wildboar
    
## Documentation

* HTML documentation: [https://isaksamsten.github.io/wildboar](https://isaksamsten.github.io/wildboar)
	
## Citation
If you use `wildboar` in a scientific publication, I would appreciate
citations to the paper:
- Karlsson, I., Papapetrou, P. Boström, H., 2016.
 *Generalized Random Shapelet Forests*. In the Data Mining and
 Knowledge Discovery Journal
  - `ShapeletForestClassifier`

- Isak Samsten, 2020. isaksamsten/wildboar: wildboar (Version 1.0.3). Zenodo. doi:10.5281/zenodo.4264063
  - `ShapeletForestRegressor`
  - `ExtraShapeletForestClassifier`
  - `ExtraShapeletForestRegressor`
  - `IsolationShapeletForest`
