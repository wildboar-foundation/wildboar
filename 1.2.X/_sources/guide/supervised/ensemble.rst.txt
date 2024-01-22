.. currentmodule:: wildboar

###################
Ensemble estimators
###################

****************
Shapelet forests
****************

Shapelet forests, implemented in :class:`ensemble.ShapeletForestClassifier` and
:class:`ensemble.ShapeletForestRegressor`, construct ensembles of shapelet tree
classifiers or regressors respectively. For a large variety of tasks, these
estimators are excellent baseline methods.

*****************
Proximity forests
*****************

Test :class:`ensemble.ProximityForestClassifier` is an ensemble of highly randomized
Proximity Trees. Whereas conventional decision trees branch on attribute values,
and shapelet trees on distance thresholds, Proximity Trees is k-branching tree
that branches on proximity of time series to one of k pivot time series.
