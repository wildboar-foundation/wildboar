# Changelog

## [Unreleased]

### Added
* Add parameter `collection` to `datasets.load_datasets` and `list_datasets`
* Add parameter `preprocess` to `datasets.load_dataset`
* Add `datasets.preprocess`
* Add `datasets.list_collections`
* Add `model_selection.outlier.RepeatedOutlierSplit` to cross-validate
  outlier detection algorithms
* Add `liner_model.RocketClassifier
* Add `wildboar.embed` with `RandomShapeletEmbedding` and `RocketEmbedding`
* Add `tree.RocketTreeClassifier`
* For implementors:
  * Add `FeaturEngineer` to `*TreeBuilder` to support different
    feature types
  * Add `_utils.random_normal`

### Fixed

* Fix bug in `filter` of `datasets.load_datasets`
* Fix the number of outliers when setting `n_outliers` to `None` for
  `KMeansLabeler`
* Fix the number of outliers when setting `n_outliers` to `float` for
  `MinorityLabeler`

### Changed

* Rename `datasets._filter` to `datasets.filter`
* Parameter `shapelets` of `Tree` is changed to `features`
* For implementors
  * `ShapeletTreeBuilder` is renamed to `TreeBuilder`
  * Data handling has been refactored to `_data`

## [v1.0.7]

### Added

* Add `filter` keyword-argument to `datasets.load_datasets`

## [v1.0.6]

### Fixed

* Fix bug in source distribution

## [v1.0.5]

### Added

* Add `MajorityLabeler` to construct synthetic outlier datasets

### Fixed

* Fix bug in `IsolationShapeletForest` where `max_depth` was incorrectly computed

## [v1.0.4]

### Added
* Model agnostic counterfactual explanations has been added.
* Shapelet forest counterfactual explanations has been refined.
* KNearestNeighbors counterfactual explanations has been refined.
* Synthetic generation of outlier detection datasets.
* IsolationShapeletForest has been added. A novel semi-supervised method for detecting
  time series outliers.
* Fast computation of scaled and unscaled dynamic time warping (using the UCRSuite algorithm).
* LB_Keogh lower bound and envelope.
* Add new class `Repository` which represents a collection of bundles
* `datasets.set_cache_dir` to globally change the default cache directory
* `datasets.clear_cache` to clear the cache

### Deprecated

* `datasets.load_all_datasets` has been replaced by `load_datasets`

### Changed

* `wildboar.datasets.install_repository` now installs a repository instead of a bundle
* Rename `Repository` to `Bundle`

## [v1.0.3]

### Added

* Added a counterfactual explainability module

## [v1.0]

### Fixed

* Stability

## [v0.3.4]

## Changed

* Complete rewrite of the shapelet tree representation to allow releasing GIL.
  The prediction of trees should be backwards compatible, i.e., trees fitted using
  the new versions are functionally equivalent to the old but with another internal
  representation.

## [v0.3.1]

### Fixed

* Improved caching of lower-bound for DTW 
  The DTW subsequence search implementation has been improved by caching
  DTW lower-bound information for repeated calls with the same
  subsequece. This slightly increases the memory requirement, but can
  give significantly improved performance under certain circumstances.
 
* Allow shapelet information to be extracted 
  A new attribute `ts_info` is added to `Shapelet` (which is accessible 
  from `tree.root_node_.shapelet`). `ts_info` returns a tuple
  `(ts_index, ts_start, length)` with information about the index (in 
  the `x` used to fit, `fit(x, y)`, the model) and the start position of 
  the shapelet. For a shapelet tree/forest fit on `x` the shapelet in a 
  particular node is given by `x[ts_index, ts_start:(ts_start + length)]`.
  
## [v0.3]

### Added
* Regression shapelet trees 
  A new type of shapelet trees has been added. 
  `wildboar.tree.ShapeletTreeRegressor` which allows for constructing shapelet
  trees used to predict real value outputs.

* Regression shapelet forest
  A new tyoe of shapelet forest has been added. 
  `wildboar.ensemble.ShapeletForestRegressor` which allows for constructing
  shapelet forests for predicting real value outputs.

### Fixed

 * a6f656d Fix bug for strided labels not correctly accounted for
 * 321a04d Remove unused property `unscaled_threshold`
