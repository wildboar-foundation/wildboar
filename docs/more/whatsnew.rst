==========
What's new
==========

.. currentmodule:: wildboar

.. _whats-new:

.. include:: defs.hrst

Dependencies
============

Wildboar 1.3 requires Python 3.12+, numpy 1.24.1+, scipy 1.10.0+ and scikit-learn 1.8+.


Version 1.3.0
=============

**In development**

New and changed models
----------------------

Wildboar 1.3 introduces several new models.

- :class:`explain.counterfactual.NativeGuideCounterfactual` a baseline
  counterfactual explainer as proposed by Delaney et al. (2021).

- :class:`explain.counterfactual.PrototypeCounterfactual` a novel prototype
  guided counterfactual explainer as proposed by Samsten (2026).

- :class:`explain.counterfactual.NiceCounterfactual` a nearest-neighbor based
  counterfactual explainer.

- :class:`~wildboar.linear_model.CastorRegressor` for regression using
  competing dilated shapelets.

- :class:`~wildboar.distance.NearestNeighbors` for unsupervised
  nearest-neighbor search with time series distance metrics.

- Add a new module :mod:`wildboar.segment` for time series segmentation. The
  module contains :class:`~wildboar.segment.FlussSegmenter` and the helper
  function :func:`~wildboar.segment.detect_changepoints`.

- Adds a new module :mod:`wildboar.dimension_selection` to sample a subset of
  the most important dimensions when considering multivariate time series. The
  new module contains six new selector algorithms:

  - :class:`~wildboar.dimension_selection.DistanceVarianceThreshold`: remove
    dimensions where the pairwise distances has a variance below the threshold.
  - :class:`~wildboar.dimension_selection.SequentialDimensionSelector`: remove
    dimensions sequentially by adding (or removing) dimensions greedily.
  - :class:`~wildboar.dimension_selection.SelectDimensionPercentile`: only retain
    the specified fraction of dimensions with the highest score.
  - :class:`~wildboar.dimension_selection.SelectDimensionTopK`: only retain
    the specified top `k` dimensions with the highest score.
  - :class:`~wildboar.dimension_selection.SelectDimensionSignificance`: only
    retain the dimensions with a `p-value` below the specified alpha level.
  - :class:`~wildboar.dimension_selection.ECSSelector`: Select time series
    dimensions based on the sum of distances between pairs of classes.

- Add a new transform :class:`~wildboar.transform.QuantTransform` as proposed by
  Dempster et.al (2024).

- Add a new transform :class:`~wildboar.transform.FftTransform` for discrete
  Fourier transform.

- Add support for scikit-learn style preprocessing transformers:
  :class:`~wildboar.datasets.preprocess.Standardize`,
  :class:`~wildboar.datasets.preprocess.MinMaxScale`,
  :class:`~wildboar.datasets.preprocess.MaxAbsScale`,
  :class:`~wildboar.datasets.preprocess.Truncate`,
  :class:`~wildboar.datasets.preprocess.Interpolate`.

- Add a new module :mod:`~wildboar.distance.lb` for distance lower bounds. The module
  contains four lower bounds.

  - :class:`~wildboar.distance.lb.DtwKeoghLowerBound`: lower bounds dynamic time warping.
  - :class:`~wildboar.distance.lb.DtwKimLowerBound`: lower bounds dynamic time warping.
  - :class:`~wildboar.distance.lb.SaxLowerBound`: lower bounds z-normalized
    Euclidean distance.
  - :class:`~wildboar.distance.lb.PaaLowerBound`: lower bounds z-normalized
    Euclidean distance.

.. grid:: 1
  :gutter: 2

  .. grid-item-card::

     :mod:`wildboar.explain.counterfactual`
     ^^^

     - |Feature| Native guide counterfactuals.

     - |Feature| Add support for searching multiple nearest neighbors in
       :class:`~wildboar.explain.counterfactual.NiceCounterfactual` using
       `n_neighbors`.

     - |API| Remove the deprecated `scoring` and `valid_scoring` parameters in
       :func:`~wildboar.explain.counterfactual.counterfactuals`. Use `proximity`
       instead of `scoring` and the returned validity mask to select successful
       counterfactuals.

     - |API| Remove support for passing `train_x` and `train_y` through
       `method_args` in :func:`~wildboar.explain.counterfactual.counterfactuals`.
       Use the `train_x` and `train_y` keyword parameters instead.

     - |API| Remove the deprecated ``target="auto"`` option in counterfactual
       explainers. Use ``target="predict"`` instead.

  .. grid-item-card::

     :mod:`wildboar.explain`
     ^^^

     - |Enhancement| Add support for interval depth, coverage probability and
       variability in :class:`~wildboar.explain.IntervalImportance`.

     - |Fix| Correct the per-timestep averaging in
       :class:`~wildboar.explain.IntervalImportance`.

     - |Enhancement| Add support for configurable frequency-spectrum analysis
       in :class:`~wildboar.explain.FrequencyImportance`.

     - |Enhancement| Add support for coverage probability, variability and
       top-k shapelet matching in :class:`~wildboar.explain.ShapeletImportance`.
       Matching can be controlled using `k_best`, `score_func` and `alpha`.

  .. grid-item-card::

     :mod:`wildboar.datasets`
     ^^^

     - |Fix| Increase the timeout upon first use. We also issue a better error
       message to alert the user to use
       :func:`~wildboar.datasets.refresh_repositories` to refresh the
       repositories.

     - |Fix| Avoid division by zero in :func:`~wildboar.datasets.preprocess.standardize`.

     - |Feature| Add support for interpolating missing values. Support is
       implemented for both :meth:`~wildboar.datasets.load_dataset` (with
       ``preprocess="interpolate"``) and as a scikit-learn compatible
       transformer :class:`~wildboar.datasets.preprocess.Interpolate`

     - |Feature| Add support for scikit-learn compatible transformers for
       `standardize`, `minmax_scale`, `maxabs_scale` and `truncate`.

     - |Fix| Correctly apply callable filters in
       :func:`~wildboar.datasets.load_datasets`.

  .. grid-item-card::

     :mod:`wildboar.distance`
     ^^^

     - |API| Rename `matrix_profile` to `paired_matrix_profile` and issue a
       deprecation warning in `matrix_profile`. The new function reverses
       the meaning of X and Y, i.e., annotate every subsequence in X with the
       closest match in Y (instead of the reverse).

     - |Feature| A new function :func:`~wildboar.distance.matrix_profile` for
       computing the matrix profile for every subsequence in all time series.
       By default it will raise a deprecation warning and delegate to
       `paired_matrix_profile` (until 1.4), after which the `kind="default"` will
       be the default value. To keep the current behaviour set `kind="paired"`
       and swap the order of `X` and `Y` or use
       :func:`~wildboar.distance.paired_matrix_profile`.

     - |Feature| Add support for array-like thresholds in
       :func:`~wildboar.distance.subsequence_match`.

     - |Feature| Add a new estimator
       :class:`~wildboar.distance.NearestNeighbors` for nearest-neighbor search.

     - |API| Change the default value of `dim` in distance functions from
       ``"warn"`` to ``"mean"``. By default, distances are averaged over the
       dimensions of multivariate time series.

     - |API| Rename the boolean `metric` parameter in
       :class:`~wildboar.distance.MDS` to `metric_mds`. Rename `dissimilarity`
       and `dissimilarity_params` to `metric` and `metric_params`, respectively.

     - |API| Change the default value of `normalized_stress` in
       :class:`~wildboar.distance.MDS` to ``"auto"``.

  .. grid-item-card::

     :mod:`wildboar.linear_model`
     ^^^

     - |Enhancement| Add parameter validation to all models in the
       :mod:`wildboar.linear_model` module.

     - |Feature| Add :class:`~wildboar.linear_model.CastorRegressor` for
       regression using competing dilated shapelets.

  .. grid-item-card::

     :mod:`wildboar.segment`
     ^^^

     - |Feature| Add :class:`~wildboar.segment.FlussSegmenter` for matrix
       profile based time series segmentation.

     - |Feature| Add :func:`~wildboar.segment.detect_changepoints` for
       changepoint detection using PELT or binary segmentation.

     - |API| Deprecate :func:`~wildboar.annotate.segment` which will be removed
       in 1.4. Use :class:`~wildboar.segment.FlussSegmenter` instead.

  .. grid-item-card::

     :mod:`wildboar.ensemble`
     ^^^

     - |Fix| Correctly use `n_jobs` in
       :class:`~wildboar.ensemble.ElasticEnsembleClassifier`.

  .. grid-item-card::

     :mod:`wildboar.tree`
     ^^^

     - |Fix| Reset the split criterion between candidate shapelets when
       fitting shapelet trees. Previously, the first shapelet was selected
       irrespective of the number of candidates, effectively treating
       `n_shapelets` as 1. The fix also affects shapelet forests.

     - |Feature| Add support for dictionary-based metric specifications in
       :class:`~wildboar.tree.ProximityTreeClassifier`.

     - |API| Remove the deprecated ``metric="default"`` option in
       :class:`~wildboar.tree.ProximityTreeClassifier`.

     - |Feature| Add a new hyper-parameter `impurity_equality_tolerance` which
       controls how we treat impurities as equal. If the impurity of two
       shapelets are the same we consider the separation gap. By default the
       distance separation gap is disabled
       (``impurity_equality_tolerance=None``) but it can be enabled by setting
       a (small) non negative float.

     - |Feature| Add support for plotting decision trees using the
       :func:`~wildboar.tree.plot_tree` function.

     - |Feature| Add support for different strategies when constructing
       shapelet trees. When `strategy="best"`, we use the matrix profile
       to find the best shapelets per sample in the sizes determined by the
       `shapelet_size` parameter. We can tune the trade-off between accuracy
       and computational cost by setting the `sample_size` parameter. The tree
       defaults to `strategy="random"` to retain backward compatibility. The
       default value will change to `strategy="best"` in 1.4 and we issue a
       deprecation warning.

     - |Feature| Add support for specifying a coverage probability instead
       of a shapelet size range using `coverage_probability` and `variability`.

     - |Enhancement| Ensure that we consider the number of dimensions when
       computing the default value for n_shapelets.


  .. grid-item-card::

     :mod:`wildboar.transform`
     ^^^

     - |API| Remove the deprecated ``intervals="sample"`` option in
       interval-based transformations. To sub-sample intervals, use
       ``intervals="fixed"`` and set `sample_size` to a float.

     - |API| Remove the deprecated ``n_intervals="log"`` option in
       interval-based transformations. Use ``n_intervals="log2"`` instead.

     - |API| Deprecate :class:`~wildboar.transform.RandomShapeletTransform`
       which will be removed in 1.4. Use
       :class:`~wildboar.transform.ShapeletTransform` with `strategy="random"`
       to keep the current behavior after 1.4.

     - |Feature| Add a new class :class:`~wildboar.transform.ShapeletTransform`
       that accept an additional parameter `strategy` which can be set to
       `"random"` or `"best"`. If set to `"best"` we use the matrix profile to
       find the best
       shapelets per sample to use in the transformation. The shapelet size is
       determined by the `shapelet_size` parameter.

     - |Feature| Add a new option to
       :class:`~wildboar.transform.DerivativeTransform` ``method`` to select
       the way in which the difference is computed. Supported options include:
       `backward`, `central`, and `slope`.

     - |API| Deprecate the ``estimate`` parameter in
       :class:`~wildboar.transform.SAX`. The same behavior can be had using
       ``scale=True`` (default).

     - |Enhancement| Add support for dyadic intervals in
       :class:`~wildboar.transform.IntervalTransform`.

     - |Enhancement| Add support for quantile summarizer in
       :class:`~wildboar.transform.IntervalTransform`.

     - |Enhancement| Add support for summarizer parameters in
       :class:`~wildboar.transform.IntervalTransform`.

     - |Feature| Add support for specifying a coverage probability instead of a
       minimum and maximum interval size in
       :class:`~wildboar.transform.IntervalTransform` when using
       `intervals="random"`.

     - |Feature| Add support for the phase spectrum in
       :class:`~wildboar.transform.FftTransform`.

     - |API| Change the default value of `n_jobs` in fixed shapelet
       transformations from -1 to 1.

     - |Fix| Fix memory leaks in interval and attribute generation.

  .. grid-item-card::

     :mod:`wildboar.utils`
     ^^^

     - |API| Remove the deprecated `iseos` function. Use
       :func:`~wildboar.utils.variable_len.is_end_of_series` instead.
