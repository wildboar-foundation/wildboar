</p>
<p align="center">
<img src="https://github.com/isaksamsten/wildboar/blob/master/.github/github-logo.png?raw=true" alt="Wildboar logo" width="100px">
</p>

<h1 align="center">wildboar</h1>

<p align="center">
	<img src="https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9-blue" />
	<img src="https://github.com/isaksamsten/wildboar/workflows/Build,%20test%20and%20upload%20to%20PyPI/badge.svg"/>
	<a href="https://badge.fury.io/py/wildboar"><img src="https://badge.fury.io/py/wildboar.svg" /></a>
	<a href="https://pepy.tech/project/wildboar"><img src="https://static.pepy.tech/personalized-badge/wildboar?period=total&units=international_system&left_color=black&right_color=orange&left_text=downloads" /></a>
	<a href="https://doi.org/10.5281/zenodo.4264063"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4264063.svg" /></a>
</p>

[wildboar](https://isaksamsten.github.io/wildboar/) is a Python module for temporal machine learning and fast
distance computations built on top of
[scikit-learn](https://scikit-learn.org) and [numpy](https://numpy.org)
distributed under the GNU Lesser General Public License Version 3. Learn more about [what's new](https://isaksamsten.github.io/wildboar/master/more/whatsnew.html).

It is currently maintained by Isak Samsten

## Features
| **Data**                                                                          | **Classification**                                                                                                                | **Regression**                  | **Explainability**                                                                                                                                        | **Metric** | **Unsupervised**                                                                                                            | **Outlier**                                                                                                               |
|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [Repositories](https://isaksamsten.github.io/wildboar/master/guide/datasets.html) | ``ShapeletForestClassifier`` | ``ShapeletForestRegressor``     | ``ShapeletForestCounterfactual`` | UCR-suite  | ``ShapeletForestEmbedding``                                                                                                 | ``IsolationShapeletForest`` |
| Classification (``wildboar/ucr``)                                                 | ``ExtraShapeletTreesClassifier``                                                                                                  | ``ExtraShapeletTreesRegressor`` | ``KNearestCounterfactual``                                                                                                                                | MASS       | ``RandomShapeletEmbedding``                                                                                                 |                                                                                                                           |
| Regression (``wildboar/tsereg``)                                                  | ``RocketTreeClassifier``                                                                                                          | ``RocketRegressor``             | ``PrototypeCounterfactual``                                                                                                                               |            | ``RocketEmbedding``                                                                                                         |                                                                                                                           |
| Outlier detection (``wildboar/outlier:easy``)                                     | ``RocketClassifier``                                                                                                              | ``RandomShapeletRegressor``     | ``IntervalImportance``                                                                                                                                    |            | ``IntervalEmbedding``                                                                                                       |                                                                                                                           |
|                                                                                   | ``RandomShapeletClassifier``                                                                                                      | ``RocketTreeRegressor``         |                                                                                                                                                           |            | ``FeatureEmbedding``                                                                                                        |                                                                                                                           |
|                                                                                   | ``RockestClassifier``                                                                                                             | ``RockestRegressor``            |                                                                                                                                                           |            | MatrixProfile               |                                                                                                                           |
|                                                                                   | ``IntervalTreeClassifier``                                                                                                        | ``IntervalTreeRegressor``       |                                                                                                                                                           |            | Segmentation |                                                                                                                           |
|                                                                                   | ``IntervalForestClassifier``                                                                                                      | ``IntervalForestRegressor``     |                                                                                                                                                           |            | Motif discovery         |                                                                                                                           |
|                                                                                   | ``ProximityTreeClassifier``                                                                                                       |                                 |                                                                                                                                                           |            |                                                                                                                             |                                                                                                                           |
|                                                                                   | ``ProximityForestClassifier``                                                                                                     |                                 |                                                                                                                                                           |            |                                                                                                                             |                                                                                                                           |

See the [documentation](https://isaksamsten.github.io/wildboar/master/examples.html) for examples.

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
