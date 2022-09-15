.. _older-versions:

==============
Older versions
==============

.. include:: defs.hrst

Dependencies
============

Wildboar 0.3 requires Python 3.6+, numpy 1.17.3+, scipy 1.3.2+ and scikit-learn 0.24+.

Version 0.3.4
=============

Changelog
---------

- |Enhancement| Complete rewrite of the shapelet tree representation to allow releasing GIL.
  The prediction of trees should be backwards compatible, i.e., trees fitted using
  the new versions are functionally equivalent to the old but with another internal
  representation.

Version 0.3.1
=============

Changelog
---------

- |Enhancement| Improved caching of lower-bound for DTW 
  The DTW subsequence search implementation has been improved by caching
  DTW lower-bound information for repeated calls with the same
  subsequece. This slightly increases the memory requirement, but can
  give significantly improved performance under certain circumstances.
 
- |Enhancement| Allow shapelet information to be extracted 
  A new attribute `ts_info` is added to `Shapelet` (which is accessible 
  from `tree.root_node_.shapelet`). `ts_info` returns a tuple
  `(ts_index, ts_start, length)` with information about the index (in 
  the `x` used to fit, `fit(x, y)`, the model) and the start position of 
  the shapelet. For a shapelet tree/forest fit on `x` the shapelet in a 
  particular node is given by `x[ts_index, ts_start:(ts_start + length)]`.

Version 0.3
===========

Changelog
---------


- |Feature| A new type of shapelet trees has been added. 
  `wildboar.tree.ShapeletTreeRegressor` which allows for constructing shapelet
  trees used to predict real value outputs.

- |Feature| A new type of shapelet forest has been added. 
  `wildboar.ensemble.ShapeletForestRegressor` which allows for constructing
  shapelet forests for predicting real value outputs.

- |Fix| Where strided labels were not correctly accounted for

- |API| Remove unused property `unscaled_threshold`