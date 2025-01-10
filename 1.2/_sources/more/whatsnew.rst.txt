==========
What's new
==========

.. currentmodule:: wildboar

.. _whats-new:

.. include:: defs.hrst

Dependencies
============

Wildboar 1.2 requires Python 3.8+, numpy 1.19.5+, scipy 1.6.0+ and scikit-learn 1.3+.

Version 1.2.1
=============

Limit scikit-learn to a version before 1.6 and build for Python 3.13.

Version 1.2.0
=============

New and changed models
----------------------

Wildboar 1.2 introduces several new models.

- :class:`transform.HydraTransform`: a new convolution based dictionary
  transformation method as described by Dempster et al., (2023).
- :class:`linear_model.HydraClassifier`: a new convolution based dictionary
  classifier as described by Dempster et al., (2023).
- :class:`distance.KNeighborsClassifier`: the traditional k-neighbors
  classifier using wildboar native distance metrics, including the full suite
  of optimized elastic metrics.
- :class:`distance.KMeans`: the traditional k-means clustering algorithm.
  Compared to scikit-learn, this implementation supports `dtw` and `wdtw`.
- :class:`distance.KMedoids`: the traditional k-medoids clustering algorithm
  with support for all elastic metrics.
- :class:`ensemble.ElasticEnsembleClassifier`: the elastic ensemble classifier
  as described by Lines and Bagnall (2015).
- :class:`transform.DilatedShapeletTransform`: a new shapelet based transform
  as described by Guillaume et al., (2022).
- :class:`linear_model.DilatedShapeletClassifier`: a new shapelet based
  classifier as described by Guillaume et al., (2022).
- :class:`transform.CastorTransform`: a new shapelet based transform using
  competing shapelets introduced in Samsten and Lee (2024).
- :class:`linear_model.CastorClassifier`: a new shapelet based classifier using
  competing shapelets introduced in Samsten and Lee (2024).

Changelog
---------

.. grid:: 1

  .. grid-item-card::

     :mod:`wildboar.datasets`
     ^^^

     - |API| Drop support for specifying a dataset version in ``load_dataset``.
       Support was dropped to ensure consistency between the repository
       declaration of arrays and what is available in the downloaded bundle.

     - |Fix| Correctly detect duplicate repositories.

     - |Fix| Defer repository refresh to first use.

  .. grid-item-card::

     :mod:`wildboar.distance`
     ^^^

     - |Enhancement| Improve support for 3darrays in
       :func:`distance.pairwise_distance` and :func:`distance.paired_distance`.
       By setting ``dim='mean'``, the mean distance over all dimensions are
       computed and by setting ``dim='full'`` the distance over all dimensions
       are returned. The default value for ``dim`` will change to "mean" in
       1.3. For 3darrays, we issue a deprecation warning for the current
       default value.

     - |Enhancement| Add support for standardizing all subsequence metrics.
       Prefix the name of the metric with ``"scaled_"`` to use the standardized
       metric, e.g., `"scaled_euclidean"` or `"scaled_msm"`.

     - |Enhancement| Add support for callable metrics. To support
       standardizing, we introduce new keyword parameter that pairs with the
       `"metric"` parameter called `scale` that, if set to `True` scale all
       subsequences before applying the metric. The effect of setting
       `scale=True` is the same as passing a scaled metric, e.g.,
       "scaled_euclidean".

     - |Feature| A new function :func:`distance.argmin_distance` which takes as
       input two arrays `X` and `Y` and finds, for each sample in `X`, the
       indices of the `k` samples in `Y` with the smallest distance to the i:th
       sample in `X`.

     - |Feature| A new function :func:`distance.distance_profile` which takes a
       subsequence `Y` and a (collection) of time series `X` and returns the
       distance from `Y` to all subsequences of the same length of the i:th
       sample in `X`, with support for `dilation` and `padding`.

     - |Feature| A new function :func:`distance.argmin_subsequence_distance`
       which takes two paired arrays of subsequences and samples and finds the
       `k` smallest matching positions for each sample/subsequence pair.

     - |Feature| Enables support for ``n_jobs`` in
       :func:`distance.pairwise_distance`, :func:`distance.paired_distance`.

     - |Feature| Add support for Amercing Dynamic Time Warping (subsequence)
       distance.

     - |Feature| Add support for LCSS subsequence distance.

     - |Feature| Add support for EDR subsequence distance.

     - |Feature| Add support for TWE subsequence distance.

     - |Feature| Add support for MSM subsequence distance.

     - |Feature| Add support for ERP subsequence distance.

     - |Fix| Fix a bug in angular distance leading to ``NaN`` values.

     - |Fix| Fix a bug in angular distance subsequence matching where an incorrect
       threshold was set.

     - |Fix| Fix the return value of :func:`distance.paired_distance` to
       `(n_dims, n_samples)` when `dim="full"`.

     - |API| Rename LCSS ``threshold`` parameter to ``epsilon``. We will remove
       ``threshold`` in 1.4.

     - |API| Rename EDR ``threshold`` parameter to ``epsilon``. We will remove
       ``threshold`` in 1.4.

     - |API| Rename :class:`_distance.DistanceMeasure` to ``Metric`` and
       :class:`_distance.SubsequenceDistanceMeasure` to ``SubsequenceMetric``.
       The change only affect code that ``cimport`` modules.

     - |API| The default value of `threshold` in
       :func:`distance.subsequence_match` has changed to `"auto"`. The old
       value `"best"` has been deprecated and will be removed in 1.3.

  .. grid-item-card::

     :mod:`wildboar.ensemble`
     ^^^

     - |Feature| Add support for multiple metrics in
       :class:`ensemble.ShapeletForestClassifier`,
       :class:`ensemble.ShapeletForestRegressor`. All estimators with a
       ``metric`` parameter and which implements the ``ShapeletMixin`` are
       affected by this change.

     - |API| Rename the constructor parameter ``base_estimator`` to
       ``estimator`` in :class:`ensemble.BaggingClassifier` and
       :class:`ensemble.BaggingRegressor`. ``base_estimator`` is deprecated in
       1.2 and will be removed in 1.4.

     - |API| Change the tuple argument for ``kernel_size`` to two new
       parameters ``min_size`` and ``max_size``. This change affect
       :class:`tree.RocketForestClassifier` and
       :class:`tree.RocketForestRegressor`.

     - |Fix| Fix a bug where ``sampling`` was incorrectly set for
       :class:`ensemble.RocketForestClassifier` and
       :class:`ensemble.RocketForestRegressor`.

     - |API| Change the default value of ``n_shapelets`` to "log2" for
       :class:`ensemble.ShapeletForestClassifier` and
       :class:`ensemble.ShapeletForestRegressor`.

     - |API| Drop support for ``criterion="mse"`` in
       :class:`ensemble.ShapeletForestRegressor` and
       :class:`ensemble.ExtraShapeletTreesRegressor`.

  .. grid-item-card::

     :mod:`wildboar.explain.counterfactual`
     ^^^

     - |Feature| Add support for KNeighborsClassifiers fitted with any metric
       in :class:`explain.counterfactual.KNeighborsCounterfactual`. We allow
       for using different methods for finding the counterfactuals for
       `n_neighbors > 1` by setting `method='mean'` or `method='medoid'`. We
       have also improved the way in which cluster centroids are selected,
       resulting in a more robust counterfactuals.

  .. grid-item-card::

     :mod:`wildboar.linear_model`
     ^^^

     - |API| Undeprecate the ``normalize`` parameter from :class:`linear_model.RocketClassifier` and
       :class:`linear_model.RocketRegressor`.

  .. grid-item-card::

     :mod:`wildboar.transform`
     ^^^

     - |Feature| Add support for multiple metrics in :class:`transform.RandomShapeletTransform`
       by passing a list of metric specifications. See the documentation for details.

     - |Enhancement| Rename the parameter value ``log`` for the parameter ``n_intervals``
       in :class:`transform.IntervalTransform` to ``log2``. The old value is deprecated
       and will be removed in 1.4.

     - |Feature| Improve the ``metric`` specification for :class:`transform.PivotTransform`.

     - |API| Change the tuple argument for ``kernel_size`` to two new parameters ``min_size`` and ``max_size``.
       This change affect :class:`transform.RocketTransform`.

  .. grid-item-card::

     :mod:`wildboar.tree`
     ^^^

     - |Feature| Add support for multiple metrics in
       :class:`tree.ShapeletTreeClassifier`,
       :class:`tree.ShapeletTreeRegressor`. All estimators with a ``metric``
       parameter and which implements the ``ShapeletMixin`` is affected by this
       change.

     - |Fix| Correctly use MSM distance measure in
       :class:`tree.ProximityTreeClassifier`.

     - |Fix| Correctly set ``min_samples_leaf`` in
       :class:`tree.RocketTreeClassifier` and :class:`RocketTreeRegressor`.

     - |API| Change the tuple argument for ``kernel_size`` to two new
       parameters ``min_size`` and ``max_size``. This change affect
       :class:`tree.RocketTreeClassifier` and
       :class:`tree.RocketTreeRegressor`.

     - |API| The ``metric_factories`` parameter of
       :class:`tree.ProximityTreeClassifier` has been renamed to ``metric``. We
       have deprecated ``metric_factories`` and it will be removed in 1.4. We
       also introduce the ``metric_params`` argument for single metric uses.

     - |API| Change the default value of ``n_shapelets`` to "log2" for
       :class:`tree.ShapeletTreeClassifier` and
       :class:`tree.ShapeletTreeRegressor`.

     - |API| Drop support for ``criterion="mse"`` in
       :class:`tree.ShapeletTreeRegressor` and
       :class:`tree.ExtraShapeletTreeRegressor`.

Other improvements
------------------

- Remove all dependencies on deprecated Numpy APIs.
- Migrate to the new scikit-learn parameter validation framework.
