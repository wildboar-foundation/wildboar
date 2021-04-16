===================
Supervised learning
===================

In supervised learning, we are given a collection of labeled time series and
the goal is to produce an estimator that as accurately as possible can map
previously unseen time series to the correct label.

Shapelet trees
==============

For classification, wildboar includes two types of shapelet trees. Both trees are based
on randomly sampling shapelets. ``ShapeletTreeClassifier`` samples shapelets
randomly and ``ExtraShapeletTreeClassifier`` also sample the distance threshold randomly.
Both ``ShapeletTreeRegressor`` and ``ExtraShapeletTreeRegressor`` are available.

Shapelet forest
===============
Shapelet forests, implemented in ``ShapeletForestClassifier`` and ``ShapeletForestRegressor``,
construct ensembles of shapelet tree classifiers or regressors respectively. For a large
variety of tasks, these estimators are excellent baseline methods.


Embedding methods
=================

``RocketClassifier`` and ``RocketRegressor`` uses a random convolutional embdding
to represent time series and fit a ridge regression model to the representation. For
benchmark tasks, this embedding and estimator configuration often give state-of-the-art
predictive performance.