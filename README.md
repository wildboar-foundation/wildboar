# wildboar
![Python version](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9-blue)
![Build, test and upload to PyPI](https://github.com/isaksamsten/wildboar/workflows/Build,%20test%20and%20upload%20to%20PyPI/badge.svg)
[![PyPI version](https://badge.fury.io/py/wildboar.svg)](https://badge.fury.io/py/wildboar)
[![Downloads](https://static.pepy.tech/personalized-badge/wildboar?period=total&units=international_system&left_color=black&right_color=orange&left_text=downloads)](https://pepy.tech/project/wildboar)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4264063.svg)](https://doi.org/10.5281/zenodo.4264063)

[wildboar](https://isaksamsten.github.io/wildboar/) is a Python module for temporal machine learning and fast
distance computations built on top of
[scikit-learn](https://scikit-learn.org) and [numpy](https://numpy.org)
distributed under the GNU Lesser General Public License Version 3.

It is currently maintained by Isak Samsten

## Features
| **Data**                                                                          | **Classification**                                                                                                                | **Regression**                  | **Explainability**                                                                                                                                        | **Metric** | **Unsupervised**                                                                                                            | **Outlier**                                                                                                               |
|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [Repositories](https://isaksamsten.github.io/wildboar/master/guide/datasets.html) | [``ShapeletForestClassifier``](https://isaksamsten.github.io/wildboar/master/examples/supervised.html#univariate-shapelet-forest) | ``ShapeletForestRegressor``     | [``ShapeletForestCounterfactual``](https://isaksamsten.github.io/wildboar/master/examples/counterfactuals.html#comparison-of-counterfactual-explanations) | UCR-suite  | ``ShapeletForestEmbedding``                                                                                                 | [``IsolationShapeletForest``](https://isaksamsten.github.io/wildboar/master/examples/unsupervised.html#outlier-detection) |
| Classification (``wildboar/ucr``)                                                 | ``ExtraShapeletTreesClassifier``                                                                                                  | ``ExtraShapeletTreesRegressor`` | ``KNearestCounterfactual``                                                                                                                                | MASS       | ``RandomShapeletEmbedding``                                                                                                 |                                                                                                                           |
| Regression (``wildboar/tsereg``)                                                  | ``RocketTreeClassifier``                                                                                                          | ``RocketRegressor``             | ``PrototypeCounterfactual``                                                                                                                               |            | ``RocketEmbedding``                                                                                                         |                                                                                                                           |
| Outlier detection (``wildboar/outlier:easy``)                                     | ``RocketClassifier``                                                                                                              | ``RandomShapeletRegressor``     | ``IntervalImportance``                                                                                                                                    |            | ``IntervalEmbedding``                                                                                                       |                                                                                                                           |
|                                                                                   | ``RandomShapeletClassifier``                                                                                                      | ``RocketTreeRegressor``         |                                                                                                                                                           |            | ``FeatureEmbedding``                                                                                                        |                                                                                                                           |
|                                                                                   | ``RockestClassifier``                                                                                                             | ``RockestRegressor``            |                                                                                                                                                           |            | [``matrix_profile``](https://isaksamsten.github.io/wildboar/master/examples/unsupervised.html#matrix-profile)               |                                                                                                                           |
|                                                                                   | ``IntervalTreeClassifier``                                                                                                        | ``IntervalTreeRegressor``       |                                                                                                                                                           |            | [Regime change detection](https://github.com/isaksamsten/wildboar/blob/master/examples/annotate/motif%20and%20regime.ipynb) |                                                                                                                           |
|                                                                                   | ``IntervalForestClassifier``                                                                                                      | ``IntervalForestRegressor``     |                                                                                                                                                           |            | [Motif discovery](https://github.com/isaksamsten/wildboar/blob/master/examples/annotate/motif%20and%20regime.ipynb)         |                                                                                                                           |
|                                                                                   | ``ProximityTreeClassifier``                                                                                                       |                                 |                                                                                                                                                           |            |                                                                                                                             |                                                                                                                           |
|                                                                                   | ``ProximityForestClassifier``                                                                                                     |                                 |                                                                                                                                                           |            |                                                                                                                             |                                                                                                                           |

## Installation

### Dependencies

wildboar requires:

 * python>=3.7
 * numpy>=1.17.4
 * scikit-learn>=0.21.3
 * scipy>=1.3.2
 
Some parts of wildboar is implemented using Cython. Hence, compilation
requires:

 * cython (>= 0.29.14)


### Binaries

`wildboar` is available through `pip` and can be installed with:

    pip install wildboar

Universal binaries are compiled for GNU/Linux and Python 3.7, 3.8 and 3.9. 

### Compilation

If you already have a working installation of numpy, scikit-learn, scipy and cython,
compiling and installing wildboar is as simple as:

    pip install .
	
To install the requirements, use:

    pip install -r requirements.txt

For complete instructions see the [documentation](https://isaksamsten.github.io/wildboar/master/install.html#build-and-compile-from-source)
	

## Development

Contributions are welcome. Pull requests should be
formatted using [Black](https://black.readthedocs.io).

## Usage

```python
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.datasets import load_dataset
x_train, x_test, y_train, y_test = load_dataset("GunPoint", merge_train_test=False)
c = ShapeletForestClassifier()
c.fit(x_train, y_train)
c.score(x_test, y_test)
``` 
    
See the [tutorial](https://isaksamsten.github.io/wildboar/master/tutorial.html) for more examples.

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

- Isak Samsten, 2020. isaksamsten/wildboar: wildboar. Zenodo. doi:10.5281/zenodo.4264063
    
- Karlsson, I., Rebane, J., Papapetrou, P. et al. 
  Locally and globally explainable time series tweaking. 
  Knowl Inf Syst 62, 1671–1700 (2020)
  
  - `ShapeletForestCounterfactual`
  - `KNearestCounterfactual`
