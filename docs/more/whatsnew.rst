==========
What's new
==========

.. currentmodule:: wildboar

.. _whats-new:

.. include:: defs.hrst

Dependencies
============

Wildboar 1.2 requires Python 3.8+, numpy 1.17.3+, scipy 1.3.2+ and scikit-learn 1.2+.


Version 1.2.0
=============

**In development**

New and changed models
----------------------

Wildboar 1.2 introduces several new models.

- :class:`transform.HydraTransform`: a new convolution based dictionary
  transformation method as described by Dempster et al., (2023).
- :class:`linear_model.HydraClassifier`: a new convolution based dictionary
  classifier as described by Dempster et al., (2023).
- :class:`distance.KNeighborsClassifier`: the traditional k-neighbors classifier
  using wildboar native distance metrics, including the full suite of optimized
  elastic metrics.
- :class:`distance.KMeans`: the traditional k-means clustering algorithm. Compared to
  scikit-learn, this implementation supports `dtw` and `wdtw`.
- :class:`distance.KMedoids`: the traditional k-medoids clustering algorithm with
  support for all elastic metrics.

Changelog
---------

.. grid:: 1

  .. grid-item-card::

     :mod:`wildboar.distance`
     ^^^

     - |Enhancement| Improve support for 3darrays in :func:`distance.pairwise_distance`
       and :func:`distance.paired_distance`. By setting ``dim='mean'``, the mean distance
       over all dimensions are computed and by setting ``dim='full'`` the distance over
       all dimensions are returned. The default value for ``dim`` will change to "mean"
       in 1.3. For 3darrays, we issue a deprecation warning for the current default value.

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

     - |API| Rename LCSS ``threshold`` parameter to ``epsilon``. We will remove ``threshold`` in 1.4.

     - |API| Rename EDR ``threshold`` parameter to ``epsilon``. We will remove ``threshold`` in 1.4.

     - |API| Rename :class:`_distance.DistanceMeasure` to ``Metric`` and
       :class:`_distance.SubsequenceDistanceMeasure` to ``SubsequenceMetric``. The change
       only affect code that ``cimport`` modules.

  .. grid-item-card::

     :mod:`wildboar.ensemble`
     ^^^

     - |Feature| Add support for multiple metrics in :class:`ensemble.ShapeletForestClassifier`,
       :class:`ensemble.ShapeletForestRegressor`. All estimators with a ``metric`` parameter
       and which implements the ``ShapeletMixin`` is affected by this change.

     - |API| Rename the constructor parameter ``base_estimator`` to ``estimator`` in
       :class:`ensemble.BaggingClassifier` and :class:`ensemble.BaggingRegressor`.
       ``base_estimator`` is deprecated in 1.2 and will be removed in 1.4. Original
       change in scikit-learn.

     - |API| Change the tuple argument for ``kernel_size`` to two new parameters ``min_size`` and ``max_size``.
       This change affect :class:`tree.RocketForestClassifier` and :class:`tree.RocketForestRegressor`.

     - |Fix| Fix a bug where ``sampling`` was incorrectly set for :class:`ensemble.RocketForestClassifier`
       and :class:`ensemble.RocketForestRegressor`.

     - |API| Change the default value of ``n_shapelets`` to "log2" for :class:`ensemble.ShapeletForestClassifier`
       and :class:`ensemble.ShapeletForestRegressor`.

     - |API| Drop support for ``criterion="mse"`` in :class:`ensemble.ShapeletForestRegressor`
       and :class:`ensemble.ExtraShapeletTreesRegressor`.

  .. grid-item-card::

     :mod:`wildboar.explain.counterfactual`
     ^^^

     - |Feature| Add support for KNeighborsClassifiers fitted with any metric
       in `explain.counterfactual.KNeighborsCounterfactual`. We allow for using
       different methods for finding the counterfactuals for `n_neighbors > 1`
       by setting `method='mean'` or `method='medoid'`. We have also improved
       the way in which cluster centroids are selected, resulting in a more
       robust counterfactuals.

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

     - |Feature| Add support for multiple metrics in :class:`tree.ShapeletTreeClassifier`,
       :class:`tree.ShapeletTreeRegressor`. All estimators with a ``metric`` parameter
       and which implements the ``ShapeletMixin`` is affected by this change.

     - |Fix| Correctly use MSM distance measure in :class:`tree.ProximityTreeClassifier`.

     - |Fix| Correctly set ``min_samples_leaf`` in :class:`tree.RocketTreeClassifier` and :class:`RocketTreeRegressor`.

     - |API| Change the tuple argument for ``kernel_size`` to two new parameters ``min_size`` and ``max_size``.
       This change affect :class:`tree.RocketTreeClassifier` and :class:`tree.RocketTreeRegressor`.

     - |API| The ``metric_factories`` parameter of :class:`tree.ProximityTreeClassifier` has
       been renamed to ``metric``. We have deprecated ``metric_factories`` and it will be
       removed in 1.4. We also introduce the ``metric_params`` argument for single metric uses.

     - |API| Change the default value of ``n_shapelets`` to "log2" for :class:`tree.ShapeletTreeClassifier`
       and :class:`tree.ShapeletTreeRegressor`.

     - |API| Drop support for ``criterion="mse"`` in :class:`tree.ShapeletTreeRegressor` and
       :class:`tree.ExtraShapeletTreeRegressor`.

Other improvements
------------------

- Remove all dependencies on deprecated Numpy APIs.
- Migrate to the new scikit-learn parameter validation framework.
