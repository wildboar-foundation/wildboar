.. currentmodule:: wildboar

###################
Ensemble estimators
###################

.. _shapelet_forest:

****************
Shapelet forests
****************

Shapelet forests, implemented in :class:`ensemble.ShapeletForestClassifier` and
:class:`ensemble.ShapeletForestRegressor`, construct ensembles of shapelet tree
classifiers or regressors respectively. For a large variety of tasks, these
estimators are excellent baseline methods.

.. execute::
   :context:
   :show-return:

   from wildboar.datasets import load_gun_point
   from wildboar.ensemble import ShapeletForestClassifier

   X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)
   clf = ShapeletForestClassifier(random_state=1)
   clf.fit(X_train, y_train)

The :class:`~wildboar.ensemble.ShapeletForestClassifier` class includes the
`n_jobs` parameter, which determines the number of processor cores to be
allocated for model fitting and prediction. It is advisable to assign `n_jobs`
a value of ``-1`` to utilize all available cores.

We can get the predictions by using the `predict`-function (or the
`predict_proba`-function):

.. execute::
   :context:
   :show-return:

   clf.predict(X_test)

The accuracy of the model is given by the `score`-function.

.. execute::
   :context:
   :show-return:

   clf.score(X_test, y_test)


.. _proximity_forest:

*****************
Proximity forests
*****************

Test :class:`ensemble.ProximityForestClassifier` is an ensemble of highly
randomized Proximity Trees. Whereas conventional decision trees branch on
attribute values, and shapelet trees on distance thresholds, Proximity Trees is
`k`-branching tree that branches on proximity of time series to one of `k`
pivot time series.

.. execute::
   :context: reset
   :show-return:

   from wildboar.datasets import load_gun_point
   from wildboar.ensemble import ProximityForestClassifier

   X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)
   clf = ProximityForestClassifier(random_state=1)
   clf.fit(X_train, y_train)

By default, :class:`~wildboar.ensemble.ProximityForestClassifier` uses the
distance measures suggested in the original paper [#lucas]_. Using these
distance measures, we get the following accuracy:

.. execute::
   :context:
   :show-return:

   clf.score(X_test, y_test)

We can specify only a single metric:

.. execute::
   :context:
   :show-return:

   clf = ProximityForestClassifier(metric="euclidean", random_state=1)
   clf.fit(X_train, y_train)

This configuration gives the following accuracy:

.. execute::
   :context:
   :show-return:

   clf.score(X_test, y_test)


We can also specify more complex configurations by passing a ``dict`` or
``list`` to the metric parameter. You can :ref:`read more about metric
specification <metric_specification>` in the corresponding section.

.. execute::
   :context:
   :show-return:

   clf = ProximityForestClassifier(
      metric={
         "ddtw": {"min_r": 0.01, "max_r": 0.1},
         "msm": {"min_c": 0.1, "max_c": 100},
      },
      random_state=1
   )
   clf.fit(X_train, y_train)

This configuration gives the following accuracy:

.. execute::
   :context:
   :show-return:

   clf.score(X_test, y_test)


*****************
Elastic Ensemble
*****************

The Elastic ensemble is a classifier first described by Lines and Bagnall
(2015) [#lines]_. The ensemble consists of one `k`-nearest neighbors classifier
per distance metric, with the parameters of the metric optimized through leave
one out cross-validation.

.. execute::
   :context: reset
   :show-return:

   from wildboar.datasets import load_gun_point
   from wildboar.ensemble import ElasticEnsembleClassifier
   X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)

   clf = ElasticEnsembleClassifier()
   clf.fit(X_train, y_train)

The default configuration uses all available elastic distances measures in
Wildboar, which corresponds to a superset of the elastic metrics used by Lines
and Bagnall (2015) [#lines]_ but with a smaller grid of metric parameters.

The result of the default configuration is:

.. execute::
   :context:
   :show-return:

   clf.score(X_test, y_test)


Similar to the :ref:`Proximity Forest <proximity_forest>`, we can specify a custom metric:

.. execute::
   :context:
   :show-return:

   clf = ElasticEnsembleClassifier(
      metric={
         "ddtw": {"min_r": 0.01, "max_r": 0.1},
         "msm": {"min_c": 0.1, "max_c": 100},
      },
   )

   clf.fit(X_train, y_train)

This smaller configuration has an accuracy of:

.. execute::
   :context:
   :show-return:

   clf.score(X_test, y_test)

***************
Interval Forest
***************

The interval forest was first introduced by Deng et al. [#deng]_ and is
implemented in the class :class:`~wildboar.ensemble.IntervalForestClassifier`
It constructs a forest of interval-based decision trees where each node is
constructed using a value aggregate over a (possibly overlapping) interval. In
the default formulation a node uses either the `mean`, `variance` or `slope` of
the interval. But it is possible to consider other aggregation functions (in
Wildboar we call the functions summarization functions).

.. execute::
   :context: reset
   :show-return:

   from wildboar.datasets import load_gun_point
   from wildboar.ensemble import IntervalForestClassifier

   X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)
   clf = IntervalForestClassifier(min_size=0.1, max_size=0.3, random_state=1)
   clf.fit(X_train, y_train)

The interval forest uses the default summarization functions mentioned above
and `sqrt(n_timestep)` intervals. By default, we randomly select random
intervals that are possibly overlapping. The accuracy is:

.. execute::
   :context:
   :show-return:

   clf.score(X_test, y_test)

We can also use non-overlapping intervals by setting the `intervals` parameter
to `"fixed"`. We can sample a smaller set of intervals by setting the
`sample_size` parameter to a float.

.. warning::

   ``intervals="sample"`` was deprecated in version 1.3 and will be removed in
   version 1.4. The equivalent functionality can be achieved by setting
   ``intervals="fixed"`` and specifying `sample_size` as a float.

.. execute::
   :context:
   :show-return:

   from wildboar.datasets import load_gun_point
   from wildboar.ensemble import IntervalForestClassifier

   X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)
   clf = IntervalForestClassifier(
      intervals="fixed", n_intervals=30, sample_size=0.2, random_state=1
   )
   clf.fit(X_train, y_train)

At each node in each tree, we sample 20% of the intervals. The accuracy is:

.. execute::
   :context:
   :show-return:

   clf.score(X_test, y_test)

We can also change the summarizer. By setting the `summarizer` parameter to
``"catch22"`` we can sample from the full set of Catch22 [#lubba]_ features.

.. execute::
   :context:
   :show-return:

   X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)
   clf = IntervalForestClassifier(
      summarizer="catch22",
      intervals="random",
      n_intervals=30,
      random_state=1,
   )
   clf.fit(X_train, y_train)

Here, we sample 30 possibly overlapping intervals at each node and randomly
selects one of the catch22 features to split the node. The accuracy for this
configuration is:

.. execute::
   :context:
   :show-return:

   clf.score(X_test, y_test)

**********
References
**********

.. [#lucas] Lucas, Benjamin, Ahmed Shifaz, Charlotte Pelletier, Lachlan
   O'Neill, Nayyar Zaidi, \ Bart Goethals, François Petitjean, and Geoffrey I.
   Webb. (2019) Proximity forest: an effective and scalable distance-based
   classifier for time series. Data Mining and Knowledge Discovery

.. [#lines] Jason Lines and Anthony Bagnall, Time Series Classification with
   Ensembles of Elastic Distance Measures, Data Mining and Knowledge Discovery,
   29(3), 2015.

.. [#lubba] Lubba, C.H., Sethi, S.S., Knaute, P., Schultz, S.R., Fulcher, B.D.
   and Jones, N.S., 2019. catch22: CAnonical Time-series CHaracteristics:
   Selected through highly comparative time-series analysis. Data Mining and
   Knowledge Discovery, 33(6), pp.1821-1852.

.. [#deng] Deng, H., Runger, G., Tuv, E. and Vladimir, M., 2013. A time
   series forest for classification and feature extraction. Information
   Sciences, 239, pp.142-153.
