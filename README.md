</p>
<p align="center">
<img src="https://github.com/wildboar-foundation/wildboar/blob/master/.github/github-logo.png?raw=true" alt="Wildboar logo" width="100px">
</p>

<h1 align="center">wildboar</h1>

<p align="center">
	<img src="https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue" />
	<img src="https://github.com/wildboar-foundation/wildboar/workflows/Build,%20test%20and%20upload%20to%20PyPI/badge.svg"/>
	<a href="https://badge.fury.io/py/wildboar"><img src="https://badge.fury.io/py/wildboar.svg" /></a>
	<a href="https://pepy.tech/project/wildboar"><img src="https://static.pepy.tech/personalized-badge/wildboar?period=total&units=international_system&left_color=black&right_color=orange&left_text=downloads" /></a>
	<a href="https://doi.org/10.5281/zenodo.4264063"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4264063.svg" /></a>
</p>

[wildboar](https://wildboar.dev/) is a Python module for temporal machine learning and fast
distance computations built on top of
[scikit-learn](https://scikit-learn.org) and [numpy](https://numpy.org)
distributed under the BSD 3-Clause license.

It is currently maintained by Isak Samsten

## Features

| **Data**                                                        | **Classification**             | **Regression**                | **Explainability**             | **Metric** | **Unsupervised**             | **Outlier**               |
| --------------------------------------------------------------- | ------------------------------ | ----------------------------- | ------------------------------ | ---------- | ---------------------------- | ------------------------- |
| [Repositories](https://wildboar.dev/master/guide/datasets.html) | `ShapeletForestClassifier`     | `ShapeletForestRegressor`     | `ShapeletForestCounterfactual` | UCR-suite  | `ShapeletForestTransform`    | `IsolationShapeletForest` |
| Classification (`wildboar/ucr`)                                 | `ExtraShapeletTreesClassifier` | `ExtraShapeletTreesRegressor` | `KNearestCounterfactual`       | MASS       | `RandomShapeletEmbedding`    |                           |
| Regression (`wildboar/tsereg`)                                  | `RocketTreeClassifier`         | `RocketRegressor`             | `PrototypeCounterfactual`      | DTW        | `RocketTransform`            |                           |
| Outlier detection (`wildboar/outlier:easy`)                     | `RocketClassifier`             | `RandomShapeletRegressor`     | `IntervalImportance`           | DDTW       | `IntervalTransform`          |                           |
|                                                                 | `RandomShapeletClassifier`     | `RocketTreeRegressor`         | `ShapeletImportance`           | WDTW       | `FeatureTransform`           |                           |
|                                                                 | `RocketForestClassifier`       | `RocketForestRegressor`       |                                | MSM        | `MatrixProfileTransform`     |                           |
|                                                                 | `IntervalTreeClassifier`       | `IntervalTreeRegressor`       |                                | TWE        | Segmentation                 |                           |
|                                                                 | `IntervalForestClassifier`     | `IntervalForestRegressor`     |                                | LCSS       | Motif discovery              |                           |
|                                                                 | `ProximityTreeClassifier`      |                               |                                | ERP        | `SAX`                        |                           |
|                                                                 | `ProximityForestClassifier`    |                               |                                | EDR        | `PAA`                        |                           |
|                                                                 | `HydraClassifier`              |                               |                                | ADTW       | `MatrixProfileTransform`     |                           |
|                                                                 | `KNeighborsClassifier`         |                               |                                |            | `HydraTransform`             |                           |
|                                                                 | `ElasticEnsembleClassifier`    |                               |                                |            | `KMeans` with (W)DTW support |                           |
|                                                                 | `DilatedShapeletClassifier`    |                               |                                |            | `KMedoids`                   |                           |
|                                                                 |                                |                               |                                |            | `DilatedShapeletTransform`   |                           |

See the [documentation](https://wildboar.dev/master/) for examples.

## Installation

### Binaries

`wildboar` is available through `pip` and can be installed with:

    pip install wildboar

Universal binaries are compiled for Python 3.8, 3.9, 3.10 and 3.11 running on
GNU/Linux, Windows and macOS.

### Compilation

If you already have a working installation of numpy, scikit-learn, scipy and cython,
compiling and installing wildboar is as simple as:

    pip install .

To install the requirements, use:

    pip install -r requirements.txt

For complete instructions see the [documentation](https://wildboar.dev/master/install.html#build-and-compile-from-source)

## Usage

```python
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.datasets import load_dataset
x_train, x_test, y_train, y_test = load_dataset("GunPoint", merge_train_test=False)
c = ShapeletForestClassifier()
c.fit(x_train, y_train)
c.score(x_test, y_test)
```

The [User guide](https://wildboar.dev/master/guide.html) includes more
detailed usage instructions.

## Changelog

The [changelog](https://wildboar.dev/master/more/whatsnew.html) records a
history of notable changes to `wildboar`.

## Development

Contributions are welcome! The [developer's
guide](https://wildboar.dev/master/more/contributing.html) has detailed
information about contributing code and more!

In short, pull requests should:

- be well motivated
- be formatted using Black
- add relevant tests
- add relevant documentation

## Source code

You can check the latest sources with the command:

    git clone https://github.com/wildboar-foundation/wildboar

## Documentation

- HTML documentation: [https://wildboar.dev](https://wildboar.dev)

## Citation

If you use `wildboar` in a scientific publication, I would appreciate
citations to the paper:

- Karlsson, I., Papapetrou, P. Boström, H., 2016.
  _Generalized Random Shapelet Forests_. In the Data Mining and
  Knowledge Discovery Journal

  - `ShapeletForestClassifier`

- Isak Samsten, 2020. isaksamsten/wildboar: wildboar. Zenodo. doi:10.5281/zenodo.4264063
- Karlsson, I., Rebane, J., Papapetrou, P. et al.
  Locally and globally explainable time series tweaking.
  Knowl Inf Syst 62, 1671–1700 (2020)

  - `ShapeletForestCounterfactual`
  - `KNearestCounterfactual`
