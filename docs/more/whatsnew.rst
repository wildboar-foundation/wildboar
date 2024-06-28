==========
What's new
==========

.. currentmodule:: wildboar

.. _whats-new:

.. include:: defs.hrst

Dependencies
============

Wildboar 1.2 requires Python 3.8+, numpy 1.17.3+, scipy 1.3.2+ and scikit-learn 1.3+.


Version 1.3.0
=============

**In development**

New and changed models
----------------------

Wildboar 1.2 introduces several new models.

- :class:`explain.counterfactual.NativeGuideCounterfactual` a baseline
  counterfactual explainer as proposed by Delaney et al. (2021).

- Adds a new module :mod:`wildboar.dimension_selection` to sample a subset of
  the most important dimensions when considering mult-variate time series. The
  new module contains four new selector algorithms:

  - :class:`~wildboar.dimension_selection.DistanceVarianceThreshold`: remove
    dimensions where the pairwise distances has a variance below the threshold.
  - :class:`~wildboar.dimension_selection.SequentialDimensionSelector`: remove
    dimensions sequentially by adding (or removing) dimensions greedily.
  - :class:`~wildboar.dimension_selection.SelectDimensionPercentile`: only retain
    the specified fraction of dimensions with the highest score.
  - :class:`~wildboar.dimension_selection.SelectDimensionTopK`: only retain
    the specified top `k` dimensions with the highest score.
  - :class:`~wildboar.dimension_selection.SelectDimensionPercentile`: only retain
    the dimensions with a `p-value` below the specified alpha level.
  - :class:`~wildboar.dimension_selection.ECSSelector`: Select time series
    dimensions based on the sum of distances between pairs of classes.



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

  .. grid-item-card::

     :mod:`wildboar.datasets`
     ^^^

     - |Fix| Increase the timeout upon first use. We also issue a better error
       message to alert the user to use
       :func:`~wildboar.datasets.refresh_repositories` to refresh the
       repositories.

     - |Fix| Avoid division by zero in :func:`~wildboar.datasets.preprocess.standardize`.

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

  .. grid-item-card::

     :mod:`wildboar.linear_model`
     ^^^

     - |Enhancement| Add parameter validation to all models in the
       :mod:`wildboar.linear_model` module.

  .. grid-item-card::

     :mod:`wildboar.tree`
     ^^^

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


  .. grid-item-card::

     :mod:`wildboar.transform`
     ^^^

     - |API| Deprecate the "sample" argument for `intervals` in interval-based
       transformations. To sub-sample intervals, set `sample_size` to a float.
     - |API| Deprecate :class:`~wildboar.transform.RandomShapeletTransform`
       which will be removed in 1.4. Use
       :class:`~wildboar.transform.ShapeletTransform` with `strategy="random"`
       to keep the current behavior after 1.4.
     - |Feature| Add a new class :class:`~wildboar.transform.ShapeletTransform` that
       accept an additional parameter `strategy` which can be set to `"random"`
       or `"best"`. If set to `"best"` we use the matrix profile to find the best
       shapelets per sample to use in the transformation. The shapelet size is
       determined by the `shapelet_size` parameter.
     - |API| Deprecate the ``estimate`` parameter in :class:`~wildboar.transform.SAX`.
       The same behavior can be had using ``scale=True`` (default).
