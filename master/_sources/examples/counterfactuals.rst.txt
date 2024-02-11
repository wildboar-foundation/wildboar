###########################
Counterfactual explanations
###########################

Counterfactual explanations for time series classification provide changes that
describe how a slightly different time series instance could have led to an
alternative classification outcome. Essentially, they identify the minimum
changes needed to alter the input time series, such as changing certain values
at specific time points, in order to flip the model's decision from one class
to another. These explanations can help users understand model behavior and
decision-making by highlighting critical points in time and the nature of the
data that would need to be different for a different result.

*********************************
Nearest neighbour counterfactuals
*********************************

Wildboar facilitates the generation of counterfactual explanations for time
series classified by *k*-nearest neighbors classifiers. Presently, two
algorithms are implemented for this purpose. The initial algorithm, as
delineated by Karlsson et al. [1]_, employs the arithmetic mean of the
*k*-nearest time series belonging to the contrasting class. The alternative
algorithm utilizes the medoid of the *k*-nearest time series. These algorithms
are incorporated within the class
:class:`~wildboar.explain.counterfactual.KNeighborsCounterfactual`. The
parameter ``method`` is provided to select between the two counterfactual
computation methods.

Although the approaches might appear similar, the
former is applicable exclusively to :class:`~wildboar.distance.KNeighborsClassifier`
configured with the ``metric`` parameter set to either ``dtw`` or ``euclidean``, and
to :class:`sklearn.neighbors.KNeighborsClassifier` when the ``metric`` parameter is
specified as ``euclidean`` or as ``minkowski`` with ``p=2``. The latter
approach is applicable to any metric configuration.

To generate counterfactuals, we first need to import the require classes. In
this example we will be using :class:`~wildboar.distance.KNeighborsClassifier`
and explain the classification outcome using
:class:`~wildboar.explain.counterfactual.KNeighborsCounterfactual`. We also
load :func:`~wildboar.datasets.load_dataset` to download a benchmark dataset.

.. execute::
   :context:

   from wildboar.datasets import load_dataset
   from sklearn.model_selection import train_test_split

In this example, we will make use of the ``ECG200`` dataset, which contains
electrocardiogram (ECG) signals and is used for binary classification tasks. It
contains time series data representing ECG recordings, where the goal is to
distinguish between normal heartbeat signals and those that correspond to a
particular type of abnormal cardiac condition. Each time series in the ECG200
dataset corresponds to an ECG signal, and the classes represent whether the
signal is from a normal heart or one with a specific anomaly.

.. execute::
   :context:

   X, y = load_dataset("ECG200")
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

.. execute::
   :context:
   :include-source: no
   :show-output:
   :card-width: 75%

   from wildboar.utils.plot import plot_time_domain
   n_samples, n_timestep = X_train.shape
   y_labels, counts = np.unique(y_train, return_counts=True)

   print(f"""
   The dataset contains {n_samples} samples with {n_timestep} time steps each.
   Of the samples, {counts[0]} is labeled as {y_labels[0]} and {counts[1]} labeled
   as {y_labels[1]}. Here, we plot the time series.
   """)
   plot_time_domain(X_train, y_train, cmap=None)


Next we fit a *k*-nearest neighbors classifier with five neighbors using
Dynamic Time Warping.

.. execute::
   :context: close-figs
   :show-return:

   from wildboar.distance import KNeighborsClassifier
   nn = KNeighborsClassifier(n_neighbors=5, metric="dtw", metric_params={"r": 0.5})
   nn.fit(X_train, y_train)

.. execute::
   :context:
   :show-output:
   :include-source: no

   score = nn.score(X_test, y_test)
   print(f"""
   The resulting estimator has an accuracy of {score * 100}%.
   """)

To compute counterfactuals, we utilize the
:class:`~wildboar.explain.counterfactual.KNeighborsCounterfactual` class from
the :mod:`~wildboar.explain.counterfactual` module. A counterfactual explainer
comprises two primary methods for interaction: ``fit(estimator)`` and
``explain(X, desired_label)``. The ``fit`` method requires an estimator for
which the counterfactuals are to be constructed, while the ``explain`` method
requires an array of time series to be modified and an array of the desired
labels. In the provided code example, we initially predict the labels for each
sample in ``X`` and subsequently create a new array of desired labels to
construct counterfactuals predicted as label ``-1`` for all samples initially
predicted as ``1`` and vice versa. Subsequently, we fit the counterfactual
explainer to the estimator and calculate the counterfactuals.
Since the ``method`` parameter is set to ``"auto"``, the explainer
will utilize the *k*-means algorithm and assign the nearest cluster centroid at
which the classifier is expected to predict the target class.

.. execute::
   :context: close-figs
   :show-return:

   from wildboar.explain.counterfactual import KNeighborsCounterfactual
   def find_counterfactuals(estimator, explainer, X):
      y_pred = estimator.predict(X)
      y_desired = np.empty_like(y_pred)

      # Store an array of the desired label for each sample.
      # We assume a binary classification task and the the desired
      # label is the inverse of the predicted label.
      a, b = estimator.classes_
      y_desired[y_pred == a] = b
      y_desired[y_pred == b] = a

      # Initialize the explainer, using the medoid approach.
      explainer.fit(estimator)

      # Explain each sample in X as the desired label in y_desired
      X_cf = explainer.explain(X, y_desired)
      return X_cf, y_pred, estimator.predict(X_cf)

   explainer = KNeighborsCounterfactual(random_state=1, method="auto")
   X_cf, y_pred, cf_pred = find_counterfactuals(nn, explainer, X_test)
   X_cf


We now have three arrays: ``X_cf``, ``y_pred``, and ``cf_pred``, which contain
the counterfactual samples, the predicted labels of the original samples, and
the predicted labels of the counterfactual samples, respectively. Subsequently, we will plot
the original and counterfactual samples with indices ``4`` and ``36``,
alongside the Euclidean average time series of the desired class.

.. execute::
   :context:
   :include-source: no
   :show-source-link:
   :link-text: Download plot source
   :card-width: 75%

   def plot_counterfactual(i, X_test, y_test, y_pred):
      plt.plot(
         X_test[i],
         label="original (y_pred = %d, y_actual = %d)" % (y_pred[i], y_test[i]),
         lw=0.5,
      )
      plt.plot(X_cf[i], label="counterfactual (y = %d)" % cf_pred[i], lw=0.5)
      plt.plot(
         np.mean(X_test[y_test == cf_pred[i]], axis=0),
         linestyle="dashed",
         label="mean of X with y = %d" % cf_pred[i],
         lw=0.5,
      )
      plt.legend()
      plt.title("Sample #%d" % i)

   plt.figure()
   plot_counterfactual(4, X_test, y_test, y_pred)
   plt.figure()
   plot_counterfactual(15, X_test, y_test, y_pred)
   plt.figure()
   plot_counterfactual(36, X_test, y_test, y_pred)

*******************************
Shapelet forest counterfactuals
*******************************

One of the first methods for computing counterfactual explanations for time
series was proposed by Karlsson et al. (2018) [2]_ and the proposed method make
use of the random shapelet trees that are part of a random shapelet forest. In
Wildboar, the random shapelet forest is implemented in the class
:class:`~wildboar.ensemble.ShapeletForestClassifier` and we can construct
counterfactuals for a shapelet forest using the class
:class:`~wildboar.explain.counterfactual.ShapeletForestCounterfactual`.

Reusing the same dataset as for the *k*-nearest neighbors classifier, we can
fit a shapelet forest classifier.

.. execute::
   :context:
   :show-return:

   from wildboar.ensemble import ShapeletForestClassifier

   rsf = ShapeletForestClassifier(
      n_estimators=100,
      metric="euclidean",
      max_depth=5,
      random_state=1,
   )
   rsf.fit(X_train, y_train)

.. execute::
   :context:
   :show-output:
   :include-source: no

   score = rsf.score(X_test, y_test)
   print(f"""
   The resulting estimator has an accuracy of {score * 100}%.
   """)

To compute counterfactuals, we use the class
:class:`~wildboar.explain.counterfactual.ShapeletForestCounterfactual` in
conjunction with the ``find_counterfactuals`` function defined previously.
Counterfactuals are generated by traversing each predictive path within the
decision trees that lead to the target outcome, and by modifying the most
closely matching shapelets in the time series to ensure that the specified
conditions are met.

.. execute::
   :context:
   :show-return:

   from wildboar.explain.counterfactual import ShapeletForestCounterfactual
   explainer = ShapeletForestCounterfactual(random_state=1)
   X_cf, y_pred, cf_pred = find_counterfactuals(rsf, explainer, X_test)
   X_cf

.. execute::
   :context:
   :include-source: no
   :show-source-link:
   :link-text: Download plot source
   :card-width: 75%

   def plot_counterfactual(i, X_test, y_test, y_pred):
      plt.plot(
         X_test[i],
         label="original (y_pred = %d, y_actual = %d)" % (y_pred[i], y_test[i]),
         lw=0.5,
      )
      plt.plot(X_cf[i], label="counterfactual (y = %d)" % cf_pred[i], lw=0.5)
      plt.plot(
         np.mean(X_test[y_test == cf_pred[i]], axis=0),
         linestyle="dashed",
         label="mean of X with y = %d" % cf_pred[i],
         lw=0.5,
      )
      plt.legend()
      plt.title("Sample #%d" % i)

   plot_counterfactual(4, X_test, y_test, y_pred)

**********
References
**********

.. [1] Karlsson, I., Rebane, J., Papapetrou, P. and Gionis, A., 2020. Locally
   and globally explainable time series tweaking. Knowledge and Information
   Systems, 62(5), pp.1671-1700.

.. [2] Karlsson, I., Rebane, J., Papapetrou, P. and Gionis, A., 2018, November.
   Explainable time series tweaking via irreversible and reversible temporal
   transformations. In 2018 IEEE International Conference on Data Mining (ICDM)
   (pp. 207-216). IEEE.
