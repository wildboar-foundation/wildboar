==========
What's new
==========

.. currentmodule:: wildboar

.. _whats-new:

.. include:: defs.hrst

Dependencies
============

Wildboar 1.1 requires Python 3.8+, numpy 1.17.3+, scipy 1.3.2+ and scikit-learn 1.1+.

Version 1.1.3
=============

Version 1.1.3 is yet another small maintenance release that adds support for
Python 3.12.

Version 1.1.2
=============

Version 1.1.2 is a small maintenance release that fixes interoperability with
recent scikit-learn releases and drops the dependency upper bounds. We also
include a small number of bug fixes.

- |Fix| Correctly interoprates with scikit-learn 1.2 and 1.3.

- |Fix| Drop upper bound on some dependencies and limit the build dependency on
  cython to `<3.0`

- |Fix| Default all wildboar estimators not allow nan.

.. grid:: 1
  
  .. grid-item-card::

     :mod:`wildboar.ensemble`
     ^^^
     
     - |Fix| Fix a bug where ``sampling`` was incorrectly set for :class:`ensemble.RocketForestClassifier`
       and :class:`ensemble.RocketForestRegressor`.

     - |Fix| Fix a bug where :class:`ensemble.ShapeletForestClassifier` does not support 
       single dimension 3D input `#74 <https://github.com/isaksamsten/wildboar/issues/74>`_. 

  .. grid-item-card::

     :mod:`wildboar.transform`
     ^^^

     - |Fix| Correctly set the minimum kernel size to 2 in :class:`tree.RocketTransform`.

  .. grid-item-card::

      :mod:`wildboar.tree`
      ^^^

      - |Fix| Correctly set ``min_samples_leaf`` in :class:`tree.RocketTreeClassifier`.
      - |Fix| Correctly set the minimum kernel size to 2 in :class:`tree.RocketTreeClassifier`.

Version 1.1.1
=============

Major changes
-------------
Correctly depend on the oldest supported Numpy release.

Changelog
---------

.. grid:: 1

  .. grid-item-card:: 
    
     :mod:`wildboar.annotate`
     ^^^

     - |Fix| Correctly return ``max_motif`` motifs from :func:`annotate.motifs`.

  .. grid-item-card:: 
    
     :mod:`wildboar.transform`
     ^^^ 

     - |Fix| Check that :class:`transform.SAX` and :class:`transform.PAA` is fitted
       before ``transform``.


Version 1.1.0
=============

New and changed models
----------------------

Wildboar 1.1 introduces a large number of new models, explanations and visualizations. Among
others we introduce :class:`explain.IntervalImportance` :class:`explain.AmplitudeImportance` 
to highlight important temporal regions; :class:`linear_model.RocketClassifier` and :class:`linear_model.RocketRegressor` 
for state-of-the-art predictive performance using random convoultions. We have also been hard 
at work adding support for :func:`distance.matrix_profile` and implemented motif and regime
search in terms of it. We also add support for three new distance metrics, weighted, derivative
and weighted derivative dynamic time warping.


Changelog
---------

.. grid:: 1

  .. grid-item-card:: 
    
     :mod:`wildboar.annotate`
     ^^^
     
     - |Feature| :func:`annotate.motifs` added to find motifs in time series

     - |Feature| :func:`annotate.segment` added to segment time series

  .. grid-item-card::

    :mod:`wildboar.datasets`
    ^^^

    - |Feature| The new module :mod:`datasets.preprocess` for preprocessing datasets.

    - |Enhancement| Setting a global cache directory has been deprecated.

    - |Enhancement| Repository definitions are cached locally to support loading cached datasets while offline.

    - |Enhancement| Support multiple preprocessing directives in :func:`datasets.load_dataset`.

    - |API| Rename ``datasets._filter`` to :mod:`datasets.filter`

  .. grid-item-card::

    :mod:`wildboar.datasets.outlier`
    ^^^

    - |API| Deprecate and remove all labelers.

    - |Enhancement| Add the new function :func:`datasets.outlier.minority_outlier`.

    - |Enhancement| Add the new function :func:`datasets.outlier.majority_outlier`.

    - |Enhancement| Add the new function :func:`datasets.outlier.kmeans_outlier`.

    - |Enhancement| Add the new function :func:`datasets.outlier.density_outlier`.

    - |Enhancement| Add the new function :func:`datasets.outlier.emmott_outlier`.
  
  .. grid-item-card:: 
    
     :mod:`wildboar.distance`
     ^^^

     - |Feature| The new :func:`distance.matrix_profile` function construct a matrix profile

     - |Feature| Support for Weighted dynamic time warping.

     - |Feature| Support for Derivative (Weighted) dynamic time warping.

     - |Feature| Support for additional Lm distance metrics: minkowski, chebyshev, manhattan.

     - |Feature| Support for cosine an angular distance.

     - |Feature| Support for LCSS, ERP, EDR, MSM, and TWE distance.

     - |Feature| The function :func:`distance.dtw.dtw_average` to compute the DTW barycenter average.
     
     - |Enhancement| The function :func:`distance.pairwise_subsequence_distance` generalizes the
       former `distance`-function and adds support for ``n_jobs``.
     
     - |Enhancement| The function :func:`distance.paired_subsequence_distance` expands the 
       use-case for the former `distance`-function.

     - |Enhancement| The function :func:`distance.pairwise_distance` adds support for full distance computation.
     
     - |Enhancement| The function :func:`distance.paired_distance` adds support for paired full distance computation.
     
     - |Enhancement| The function :func:`distance.subsequence_match` generalizes the former 
       `matches`-function and add support for `n_jobs`.
     
     - |Enhancement| The function :func:`distance.paired_subsequence_match` expands the 
       use-case for the former `matches`-function.

     - |Enhancement| The function :func:`distance.dtw.dtw_alignment` now supports the `weight` parameter for specifying optional warping weights.

     - |Feature| The new :func:`distance.dtw.wdtw_alignment` for weighted DTW alignments.

     - |API| All functions in :mod:`distance.dtw` now require keyword arguments for optional parameters.

     - |Enhancement| All DTW metrics will now default to ``r=1.0`` instead of ``r=0.0``.


  .. grid-item-card::

    :mod:`wildboar.ensemble`
    ^^^ 

    - |Feature| The new :class:`ensemble.RockestRegressor` is a classifier that constructs
      an ensemble of random convolution trees
    
    - |Feature| The new :class:`ensemble.RockestClassifier` is a regressor that constructs
      an ensemble of random convolution trees
    
    - |Feature| The new :class:`ensemble.IntervalForestClassifier` is a classifier that constructs
      an ensemble of interval trees
    
    - |Feature| The new :class:`ensemble.IntervalForestRegressor` is a regressor that constructs
      an ensemble of interval trees

    - |Feature| The new :class:`ensemble.PivotForestClassifier` is a classifier that constructs
      an ensemble of pivot trees

    - |Feature| The new :class:`ensemble.ProximityForestClassifier` is an ensemble of proximity trees.


  .. grid-item-card::

    :mod:`wildboar.transform`
    ^^^ 

    - |Feature| The new :class:`transform.RocketTransform` transform time series using random convolutions

    - |Feature| The new :class:`transform.RandomShapeletTransform` transform time series using the distance to random shapelets

    - |Feature| The new :class:`transform.IntervalTransform` transform time series using interval features

    - |Feature| The new :class:`transform.PivotTransform` transform time series using the distance to pivot time series

    - |Feature| The new :class:`transform.SAX` transforms time series into discretized bins. The implementation is
      efficient and compress the data to the smallest possible datatype.

    - |Feature| The new :class:`transform.PAA` transform time series into mean bins.

  .. grid-item-card::

    :mod:`wildboar.explain`
    ^^^ 

    - |Feature| The new :class:`explain.IntervalImportance` is an explanation that highlights
      important regions using permutation importances.

    - |Feature| The new :class:`explain.AmplitudeImportance` is an explanation that highlights
      important amplitude regions using SAX and permutation. 

    - |Enhancement| :class:`explain.counterfactual.ShapeletForestCounterfactual` now support multivariate time series.

    - |Enhancement| :class:`explain.counterfactual.ShapeletForestCounterfactual` now support the `verbose` argument.

    - |Enhancement| The `method` argument of :func:`explain.counterfactual.counterfactuals` defaults to `best`. 
      The effect is the same as for `infer`, which has been deprecated.

    - |Enhancement| :func:`explain.counterfactual.counterfactuals` now support a dictionary of `method_args`.

    - |API| The parameters `background_x` and `background_y` of :class:`explain.counterfactual.PrototypeCounterfactual`
      has been renamed to `train_x` and `train_y` and been deprecated in favor of `fit(estimator, x, y)`.

    - |API| The parameter `success` to :func:`explain.counterfactual.score` is deprecated and has been renamed to `valid`.

  .. grid-item-card::

    :mod:`wildboar.linear_model`
    ^^^ 

    - |Feature| The new :class:`linear_model.RocketClassifier` is a classifier that uses
      random convolutions and logistic regression

    - |Feature| The new :class:`linear_model.RandomShapeletClassifier` is a classifier that uses
      random shapelet distances and logistic regression

    - |Feature| The new :class:`linear_model.RocketRegressor` is a regressor that uses
      random convolutions and logistic regression

    - |Feature| The new :class:`linear_model.RandomShapeletRegressor` is a regressor that uses
      random shapelet distances and logistic regression

  .. grid-item-card::

    :mod:`wildboar.tree`
    ^^^
    
    - |Feature| The new :class:`tree.IntervalTreeClassifier` is a classifier that constructs
      trees using interval features.

    - |Feature| The new :class:`tree.IntervalTreeRegressor` is a regressor that constructs
      trees using interval features.

    - |Feature| The new :class:`tree.PivotTreeClassifier` is a classifier that uses pivot 
      time series and distance thresholds to construct trees.

    - |Feature| The new :class:`tree.ProximityTreeClassifier` is a classifier that construct
      a tree that acts as an ensemble of distance metrics.

    - |Feature| The new :class:`tree.RocketTreeClassifier` is a classifier that construct
      random convolution trees
    
    - |Feature| The new :class:`tree.RocketTreeRegressor` is a regressor that construct
      random convolution trees

    - |Enhancement| Add support for dynamically reducing the number of shapelets depending
      on the currents node depth while constructing :class:`tree.ShapeletTreeClassifier`
      and :class:`tree.ShapeletTreeRegressor`.

    - |API| Rename ``shapelet`` of the ``tree_`` attribute to ``feature``.


Other improvements
------------------

* We have improved testing, error messages and overall stability.

* Binary builds for Apple ARM processors (M1 and M2)

* Drop support for Python 3.7 and introduce support for Python 3.10
