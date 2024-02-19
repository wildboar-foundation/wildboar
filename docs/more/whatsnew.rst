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

.. grid:: 1

  .. grid-item-card::

     :mod:`wildboar.explain.counterfactual`
     ^^^

     - |Feature| Native guide counterfactuals.

  .. grid-item-card::

     :mod:`wildboar.tree`
     ^^^

     - |Feature| Add a new hyper-parameter `impurity_equality_tolerance` which
       controls how we treat impurities as equal. If the impurity of two
       shapelets are the same we consider the separation gap. By default the
       distance separation gap is disabled
       (``impurity_equality_tolerance=None``) but it can be enabled by setting
       a (small) non negative float.

