.. include:: defs.hrst
.. currentmodule:: sklearn

==========
What's new
==========

- |Feature|: something that you couldn't do before.
- |Efficiency|: an existing feature now may not require as much computation or
  memory.
- |Enhancement|: a miscellaneous minor improvement.
- |Fix|: something that previously didn't work as documentated
- |API|: you will need to change your code to have the same effect in the
  future; or a feature will be removed in the future.


.. _version-1-1:

Version 1.1
===========
**June 2022**


Dependencies
------------

Wildboar 1.1 requires Python 3.8+, numpy 1.17.3+, scipy 1.3.2+ and scikit-learn 1.1+.


New and changed models
----------------------

Wildboar 1.1 introduces a large number of new models, explanations and visualizations. Among
others we introduce :class:`IntervalImportance` to highlight important temporal regions; 
:class:`RocketClassifier` and :class:`RocketRegressor` for state-of-the-art predictive performance
using random convoultions. We have also been hard at work adding support for :func:`matrix_profile` and
implemented motif and regime search in terms of it.


Changelog
---------

.. grid:: 1

  .. grid-item-card:: 
    
     :mod:`wildboar.annotate`
     ^^^
     
     - |Feature| :func:`annotate.motifs` added to find motifs
     - |Feature| :func:`annotate.regimes`` added to find regime changes

  .. grid-item-card::

    :mod:`wildboar.datasets`
    ^^^

    - |API| Rename ``datasets._filter`` to :mod:`datasets.filter`
  
  .. grid-item-card:: 
    
     :mod:`wildboar.distance`
     ^^^

     - |Feature| The new :func:`matrix_profile` function construct a matrix profile
     
     - |Enhancement| The function :func:`distance.pairwise_subsequence_distance` generalizes the
       former `distance`-function and add support for ``n_jobs``.
     
     - |Enhancement| The function :func:`distance.paired_subsequence_distance` expands the 
       use-case for the former `distance`-function.

     - |Enhancement| The function :func:`distance.pairwise_distance` adds support for full distance computation.
     
     - |Enhancement| The function :func:`distance.paired_distance` adds support for paired full distance computation.
     
     - |Enhancement| The function :func:`distance.subsequence_match` generalizes the former 
       `matches`-function and add support for `n_jobs`.
     
     - |Enhancement| The function :func:`distance.paired_subsequence_match` expands the 
       use-case for the former `matches`-function.

  .. grid-item-card::

    :mod:`wildboar.ensemble`
    ^^^ 

    - |Feature| The new :class:`ensemble.RockestRegressor` is a classifier that constructs
      an ensemble of random convolution trees
    
    - |Feature| The new :class:`ensemble.RockestClassifier` is a regressor that constructs
      an ensemble of random convolution trees
    
    - |Feature| The new :class:`ensemble.IntervalForestClassifier` is a classifier that constructs
      an ensenble of interval trees
    
    - |Feature| The new :class:`ensemble.IntervalForestRegressor` is a regressor that constructs
      an ensenble of interval trees

  .. grid-item-card::

    :mod:`wildboar.embed`
    ^^^ 

    - |Feature| The new :class:`embed.RocketEmbedding` embed time series as random convolutions

    - |Feature| The new :class:`embed.RandomShapeletEmbedding` embed time series as the distance to random shapelets

    - |Feature| The new :class:`embed.IntervalEmbedding` embed time series as interval features

    - |Feature| The new :class:`embed.PivotEmbedding` embed time series as the distance to pivot time series

  .. grid-item-card::

    :mod:`wildboar.embed`
    ^^^ 

    - |Feature| The new :class:`explain.IntervalImportance` is an explanation that highlights
      important regions using permutation importances.

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

    - |Feature| The new :class:`tree.PivotTreeRegressor` is a regressor that uses pivot 
      time series and distance thresholds to construct trees.

    - |Feature| The new :class:`tree.ProximityTreeClassifier` is a classifier that construct
      a tree that acts as an ensemble of distance metrics.

    - |Feature| The new :class:`tree.RocketTreeClassifier` is a classifier that construct
      random convolution trees
    
    - |Feature| The new :class:`tree.RocketTreeRegressor` is a regressor that construct
      random convolution trees

    - |API| Rename ``shapelet`` of the ``tree_`` attribute to ``feature``.


Other improvements
++++++++++++++++++

* Binary builds for Apple ARM processors (M1 and M2)

* Drop support for Python 3.7 and introduce support for Python 3.10
