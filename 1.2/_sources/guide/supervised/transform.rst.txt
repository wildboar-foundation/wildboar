.. currentmodule:: wildboar


Transform-based estimators
==========================


:class:`linear_model.RocketClassifier` and
:class:`linear_model.RocketRegressor` uses a random convolutional embedding to
represent time series and fit a ridge regression model to the representation.
For benchmark tasks, this transformation and estimator configuration often give
state-of-the-art predictive performance. :class:`linear_model.HydraClassifier`
uses a similar convolutional transformation, but instead computes features
based on groups of kernels.
