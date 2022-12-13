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

     - |API| Rename LCSS ``threshold`` parameter to ``epsilon``. We will remove ``threshold`` in 1.4.

     - |API| Rename EDR ``threshold`` parameter to ``epsilon``. We will remove ``threshold`` in 1.4.

  .. grid-item-card:: 
    
     :mod:`wildboar.ensemble`
     ^^^
     
     - |API| Rename the constructor parameter ``base_estimator`` to ``estimator`` in 
       :class:`ensemble.BaggingClassifier` and :class:`ensemble.BaggingRegressor`.
       ``base_estimator`` is deprecated in 1.2 and will be removed in 1.4. Original
       change in scikit-learn.

     - |API| Change the tuple argument for ``kernel_size`` to two new parameters ``min_size`` and ``max_size``.
       This change affect :class:`tree.RocketForestClassifier` and :class:`tree.RocketForestRegressor`. 

     - |Fix| Fix a bug where ``sampling`` was incorrectly set for :class:`ensemble.RocketForestClassifier`
       and :class:`ensemble.RocketForestRegressor`.

  .. grid-item-card:: 
    
     :mod:`wildboar.linear_model`
     ^^^
     
     - |API| Remove the deprecated ``normalize`` parameter from :class:`linear_model.RocketClassifier` and
       :class:`linear_model.RocketRegressor`.

  .. grid-item-card:: 
    
     :mod:`wildboar.transform`
     ^^^

     - |Enhancement| Rename the parameter value ``log`` for the parameter ``n_intervals``
       in :class:`transform.IntervalTransform` to ``log2``. The old value is deprecated
       and will be removed in 1.4.

     - |API| Change the tuple argument for ``kernel_size`` to two new parameters ``min_size`` and ``max_size``.
       This change affect :class:`transform.RocketTransform`.

  .. grid-item-card:: 
    
     :mod:`wildboar.tree`
     ^^^
     
     - |Fix| Correctly use MSM distance measure in :class:`tree.ProximityTreeClassifier`.

     - |Fix| Correctly set ``min_samples_leaf`` in :class:`tree.RocketTreeClassifier` and :class:`RocketTreeRegressor`.

     - |API| Change the tuple argument for ``kernel_size`` to two new parameters ``min_size`` and ``max_size``.
       This change affect :class:`tree.RocketTreeClassifier` and :class:`tree.RocketTreeRegressor`.

Other improvements
------------------

- Remove all dependencies on deprecated Numpy APIs.
- Migrate to the new scikit-learn parameter validation framework.