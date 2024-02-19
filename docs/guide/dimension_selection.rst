###################
Dimension selection
###################

In multi-variate settings, it is often useful to reduce the number of
dimensions of the time series. Wildboar supports dimension selection
in the :mod:`wildboar.dimension_selection` module and implements a few
strategies inspired by traditional feature selection.


****************************
Dimension variance threshold
****************************

The simplest approach computes the variance between the pairwise distance
between time series within each dimension and is used to filter dimensions
where the time series have low or no variance.

.. execute::
   :context:
   :show-return:

   from wildboar.datasets import load_ering
   from wildboar.dimension_selection import DistanceVarianceThreshold

   X, y = load_ering()
   t = DistanceVarianceThreshold(threshold=9)
   t.fit(X, y)

We set the variance threshold to 9 to filter out any dimensions with a pairwise
distance variance greater than 9.

.. execute::
   :context:
   :show-return:

   t.get_dimensions()

The filter removes only the third dimension.

.. execute::
   :context:
   :show-return:

   t.transform(X).shape

And the resulting transformation contains only the three remaining dimensions.


*****************************
Sequential dimension selector
*****************************

Sequentially select a set of dimensions by adding (forward) or removing
(backward) dimensions to greedily form a subset. At each iteration, the
algorithm chooses the best dimension to add or remove based on the cross
validation score of a classifier or regressor.

.. execute::
   :context:
   :show-return:

   from wildboar.datasets import load_ering
   from wildboar.dimension_selection import SequentialDimensionSelector
   from wildboar.distance import KNeighborsClassifier

   X, y = load_ering()
   t = SequentialDimensionSelector(KNeighborsClassifier(), n_dims=2)
   t.fit(X, y)

We select the dimensions that have the most predictive performance.

.. execute::
   :context:
   :show-return:

   t.get_dimensions()

The resulting transformation contains only those dimensions.

.. execute::
   :context:
   :show-return:

   t.transform(X).shape

.. execute::
   :context:
   :show-return:

   from wildboar.linear_model import RocketClassifier

   X_train, X_test, y_train, y_test = load_ering(merge_train_test=False)

   clf = RocketClassifier(random_state=2)
   clf.fit(X_train, y_train)

.. execute::
   :context:
   :include-source: no
   :show-output:

   print(f"""
   Using all dimensions, the Rocket classifier has an accuracy of
   {clf.score(X_test, y_test):.2f}.
   """)


By using the :func:`~sklearn.pipeline.make_pipeline` function from ``scikit-learn``
we can reduce the number of dimensions.

.. execute::
   :context:
   :show-return:

   from sklearn.pipeline import make_pipeline
   clf = make_pipeline(
      SequentialDimensionSelector(KNeighborsClassifier(), n_dims=3),
      RocketClassifier(random_state=2)
   )
   clf.fit(X_train, y_train)

.. execute::
   :context:
   :include-source: no
   :show-output:

   print(f"""
   Using only the selected dimensions, the Rocket classifier instead has an
   accuracy of {clf.score(X_test, y_test):.2f}.
   """)


