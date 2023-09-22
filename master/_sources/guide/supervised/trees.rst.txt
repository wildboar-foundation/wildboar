.. currentmodule:: wildboar

Tree-based estimators
=====================

For classification, wildboar includes two types of shapelet trees. Both trees
are based on randomly sampling shapelets. :class:`tree.ShapeletTreeClassifier`
samples shapelets randomly and :class:`tree.ExtraShapeletTreeClassifier` also
sample the distance threshold randomly. Both
:class:`tree.ShapeletTreeRegressor` and
:class:`tree.ExtraShapeletTreeRegressor` are available.
