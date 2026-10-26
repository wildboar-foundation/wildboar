.. currentmodule:: wildboar

##########################
Transform-based estimators
##########################
Time series transformations are designed to convert time series data into
traditional column-based feature matrices suitable for input into subsequent
classifiers or regressors. Notable feature representations are Rocket and
Hydra, which employ convolutional kernels; shapelet-based transformations, which
utilize shapelet distances; and interval-based transformations, which calculate
feature values for overlapping or non-overlapping intervals. Typically these
transformations are used with a linear estimator such as
:class:`~sklearn.linear_models.RidgeClassifierCV`.

Through this section we will use the `TwoLeadECG` dataset.

.. execute::
   :context: reset

   from wildboar.datasets import load_two_lead_ecg
   X_train, X_test, y_train, y_test = load_two_lead_ecg(merge_train_test=False)

************************
Shapelet-based transform
************************
Shapelet-based transformation uses shapelets, i.e., discriminatory subsequence,
and a distance metric to construct a feature representation.

Random shapelet transform
=========================
The simplest and often effective approach is to sample a large number of
shapelets and include all without filtering in the transformation. This
approach is implementer in
:class:`~wildboar.linear_model.RandomShapeletClassifier`.

.. execute::
   :context:
   :show-return:

   from wildboar.linear_model import RandomShapeletClassifier

   clf = RandomShapeletClassifier(random_state=1)
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)

We can change the ``metric`` (by default the `metric` is set to the Euclidean
distance):

.. execute::
   :context:
   :show-return:

   clf = RandomShapeletClassifier(metric="scaled_manhattan", random_state=1)
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)


We can also specify multiple metrics, see :ref:`metric specification guide
<metric_specification>` for more information on how to format the metrics. We
can also limit the size of shapelets.

.. execute::
   :context:
   :show-return:

   clf = RandomShapeletClassifier(
      metric={"scaled_manhattan": None, "manhattan": None},
      max_shapelet_size=0.2,
      random_state=1,
   )
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)

In the example, each time series `i` is transformed into a new representation
consisting of `n` rows and `n_shapelets` features. Here, the `i`-th time series
is characterized by the minimum distance to each shapelet in the set `0, ...,
n_shapelets`. By default, each feature is normalized to have a mean of zero and
a standard deviation of one. However, this normalization can be disabled by
setting the parameter ``normalize=False``.

Dilated shapelet transform
==========================

A more recent approach, described by Guillaume et al. 2021 [#dst]_, constructs
a feature representation that incorporates not only the minimal distance but
also the occurrence counts of shapelets and the index of the minimal distance.
Consequently, each shapelet is characterized by a triad of features rather than
a singular feature. Furthermore, the Dilated Shapelet Transform (DST) expands
shapelets by inserting empty values, thereby increasing the "receptive field"
of the shapelets. This method is implemented within Wildboar as
:class:`~wildboar.linear_model.DilatedShapeletClassifier`

.. execute::
   :context:
   :show-return:

   from wildboar.linear_model import DilatedShapeletClassifier

   clf = DilatedShapeletClassifier(random_state=1)
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)

The DST classifier supports only a single `metric`; however, multiple
parameters are available for tuning. Specifically, the size of the shapelets
can be adjusted. By default, shapelets of length `7`, `9`, and `11` are
utilized. This can be modified using the parameters ``shapelet_size``,
``min_shapelet_size``, and ``max_shapelet_size``. For instance:

.. execute::
   :context:
   :show-return:

   clf = DilatedShapeletClassifier(shapelet_size=[7, 11], random_state=1)
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)

If the parameters ``min_shapelet_size`` or ``max_shapelet_size`` are specified,
all odd sizes ranging from ``n_timesteps * min_shapelet_size`` to ``n_timesteps
* max_shapelet_size`` will be utilized.

The likelihood of z-normalizing the shapelets can be adjusted by modifying the
``normalize_prob`` parameter. It defaults to `0.8`, indicating that 80 percent
of the shapelets undergo normalization.

.. execute::
   :context:
   :show-return:

   clf = DilatedShapeletClassifier(normalize_prob=0.1, random_state=1)
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)

We can also determine the occurrence threshold, that is, the threshold for
ascertaining the occurrence counts, by modifying the ``lower`` and ``upper``
parameters. These parameters delineate the bounds within which the occurrence
threshold is sampled. By default, it is sampled from the 5 to 10 percent
smallest distances.

.. execute::
   :context:
   :show-return:

   clf = DilatedShapeletClassifier(lower=0.1, upper=0.3, random_state=1)
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)


Castor transform
================

Castor (Competing diAlated Shapelet TransfORm) [#samsten]_ is a transformation
technique for time series data. Analogous to Hydra, Castor enables shapelets to
compete, and akin to DST, it utilizes the occurrence of shapelets. Castor is
characterized by two principal parameters: the number of groups (``n_groups``)
and the number of shapelets (``n_shapelets``). These parameters collectively
define the dimensions of the transformed feature space. By convention, we
employ `64` groups, each comprising `8` shapelets.

.. execute::
   :context:
   :show-return:

   from wildboar.linear_model import CastorClassifier

   clf = CastorClassifier(random_state=1)
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)

Castor has several tunable parameters, with ``n_group`` and ``n_shapelets``
being the most influential in determining classification accuracy. Generally,
increasing the values of ``n_groups`` and ``n_shapelets`` enhances accuracy, as
these parameters adjust the level of competition among features. For instance,
a configuration with ``n_groups=1`` and ``n_shapelets=1024`` results in maximal
competition, leading to a feature representation that closely resembles a
pattern dictionary. Conversely, setting ``n_groups=1024`` and ``n_shapelets=1``
eliminates competition, yielding a transformation akin to a traditional
shapelet-based transform, such as the Dilated Shapelet Transform (DST).
A recommended approach for parameter tuning is to incrementally double the
values of both ``n_group`` and ``n_shapelets`` in successive iterations.

.. warning::

   For better performance with multivariate datasets, set ``n_shapelets`` to
   `n_shapelets * n_dims` to ensure feature variability.

.. execute::
   :context:
   :show-return:

   from wildboar.linear_model import CastorClassifier

   clf = CastorClassifier(n_groups=128, n_shapelets=16, random_state=1)
   clf.fit(X_train, y_train)
   clf.score(X_test, y_test)

***************************
Convolution-based transform
***************************

Rocket
======

Hydra
=====

**********
References
**********

.. [#wistuba] Wistuba, M., Grabocka, J. and Schmidt-Thieme, L., 2015.
   Ultra-fast shapelets for time series classification. arXiv preprint
   arXiv:1503.05018.

.. [#rocket] Dempster, A., Petitjean, F. and Webb, G.I., 2020. ROCKET:
   exceptionally fast and accurate time series classification using random
   convolutional kernels. Data Mining and Knowledge Discovery, 34(5),
   pp.1454-1495.

.. [#hydra] Dempster, A., Schmidt, D.F. and Webb, G.I., 2023. Hydra: Competing
   convolutional kernels for fast and accurate time series classification. Data
   Mining and Knowledge Discovery, pp.1-27.

.. [#dst] Guillaume, A., Vrain, C. and Elloumi, W., 2022, June. Random dilated
   shapelet transform: A new approach for time series shapelets. In
   International Conference on Pattern Recognition and Artificial Intelligence
   (pp. 653-664). Cham: Springer International Publishing.

.. [#samsten] Samsten, I. and Lee, Z., 2024. Castor: Competing dilated shapelet
   transform. Forthcoming
