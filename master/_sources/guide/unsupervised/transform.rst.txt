#####################
Time series transform
#####################

***********************
Convolutional transform
***********************

Wildboar implements two convolutional transformation methods `Rocket`
[#rocket]_ and `Hydra` [#hydra]_, described by Dempsar et al. Both algorithms
employ random convolutional kernels, but in sligtly different manners. In
`Rocket`, each kernel is applied to each time series and the maximum activation
value and the average number of positive activations are recorded. In `Hydra`,
the kernels are partitioned into groups and for each exponential dilation and
padding combination each kernel is applied to each time series and the number
of times and the number of times each kernel has the highest activation value
and the lowest is recorded. Then the features corresponds to the number of
times a kernel had the in-group highest activation and the average of the
lowest activation.

For the purpose of this example, we load the `MoteStrain` dataset for the UCR
time series archive and split it into two parts: one for fitting the
transformation and one for evaluating the predictive performance.

.. execute::
   :context:

   from wildboar.datasets import load_dataset
   from sklearn.model_selection import train_test_split

   X, y = load_dataset("MoteStrain")
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


Hydra transform
===============

In Wildboar, we extensively utilize the functionalities of ``scikit-learn`` and
can directly employ these features. We construct a pipeline wherein we
initially transform each time series into the representation dictated by
`Hydra` (utilizing the default parameters ``n_groups=64`` and ``n_kernels=8``).
The subsequent stages of the pipeline include the application of a sparse
scaler, which compensates for the sparsity induced by the transformation (it is
important to note that we count the frequency of occurrences where a kernel
exhibits the highest activation, and in numerous instances, a single kernel may
never achieve this), and ultimately, the pipeline employs a standard Ridge
classifier on the transformed data.

.. execute::
   :context:
   :show-return:

   from wildboar.datasets.preprocess import SparseScaler
   from wildboar.transform import HydraTransform

   from sklearn.pipeline import make_pipeline

   hydra = make_pipeline(HydraTransform(random_state=1), SparseScaler())
   hydra.fit(X_train, y_train)

We can inspect the resulting transformation by using the ``transform`` function.

.. execute::
   :context:
   :show-return:

   X_test_transform = hydra.transform(X_test)
   X_test_transform[0]

.. execute::
   :context:
   :include-source: no
   :show-output:

   _, n_features = X_test_transform.shape
   print(f"""
   The transformed array contains {n_features} features.
   """)

We can use principal component analysis (:class:`~sklearn.decomposition.PCA`)
to identify the combination of attributes that account for most of the variance
in the data.

.. execute::
   :context:
   :include-source: no
   :show-source-link:
   :link-text: Download plot source

   import matplotlib.pylab as plt
   from sklearn.decomposition import PCA

   pca = PCA(n_components=2)
   X_test_pca = pca.fit_transform(X_test_transform)

   for label in  [1, 2]:
      plt.scatter(
         X_test_pca[y_test == label, 0],
         X_test_pca[y_test == label, 1],
         label=f"Label {label}",
      )

   plt.xlabel("Component 0")
   plt.ylabel("Component 1")
   plt.legend()

.. execute::
   :context:
   :include-source: no
   :show-output:

   evr = pca.explained_variance_ratio_
   print(f"""
      The first two components explain {(100 * evr[0]):.2f} and {(100 * evr[1]):.2f} percent of the variance.
   """)


Rocket transform
================

The Rocket transformation employs a large, randomly generated set of `kernels`
to enable the transformation process. By default, the parameter ``n_kernels``
is assigned the value of :math:`10000` kernels. Furthermore, we utilize the
pipelines offered by ``scikit-learn`` to normalize the feature representation,
ensuring a mean of zero and a standard deviation of one.

.. execute::
   :context:
   :show-return:

   from sklearn.preprocessing import StandardScaler

   from wildboar.transform import RocketTransform

   rocket = make_pipeline(RocketTransform(), StandardScaler())
   rocket.fit(X_test, y_test)

We can inspect the resulting transformation.

.. execute::
   :context:
   :show-return:

   X_test_transform = rocket.transform(X_test)
   X_test_transform[0]

In contrast to Hydra whose transformation size depends on the number of time
steps in the input, the Rocket transformation has a fixed size only dependent
on the number of kernels. As such, the resulting transformation consists of
:math:`10000` features.

We can use principal component analysis (:class:`~sklearn.decomposition.PCA`)
to identify the combination of attributes that account for most of the variance
in the data.

.. execute::
   :context:
   :include-source: no
   :show-source-link:
   :link-text: Download plot source

   pca = PCA(n_components=2)
   X_test_pca = pca.fit_transform(X_test_transform)

   for label in  [1, 2]:
      plt.scatter(
         X_test_pca[y_test == label, 0],
         X_test_pca[y_test == label, 1],
         label=f"Label {label}",
      )

   plt.xlabel("Component 0")
   plt.ylabel("Component 1")
   plt.legend()

.. execute::
   :context:
   :include-source: no
   :show-output:

   evr = pca.explained_variance_ratio_
   print(f"""
      The first two components explain {(100 * evr[0]):.2f} and {(100 * evr[1]):.2f} percent of the variance.
   """)


************************
Interval-based transform
************************
Interval-based time series transformation is a powerful technique used in time
series analysis to simplify and enhance the understanding of temporal data.
Instead of analyzing each individual time point, this method groups the data
into predefined intervals, such as days, weeks, or months, and summarizes the
information within each interval. By aggregating data in this way, we can
reduce noise and more easily identify significant patterns, trends, and
seasonal behaviors.

This approach is particularly beneficial in situations where data exhibits
periodicity, or when we need to focus on broader trends rather than detailed,
point-by-point fluctuations. By transforming time series data into
intervals, we can gain clearer insights and make more informed decisions based
on the summarized data.

We can import :class:`~wildboar.transform.IntervalTransform`:

.. execute::
   :context:

   from wildboar.transform import IntervalTransform

.. _interval_intervals:

Fixed intervals
===============
In the equally sized interval-based transformation method, the time series data
is divided into a specified number of equal-sized intervals, referred to as
``n_interval``. This approach is particularly useful when we want to analyze the
data in consistent chunks, regardless of the total duration or length of the
time series.

.. execute::
   :context:
   :show-return:

   f = IntervalTransform(intervals="fixed", n_intervals=20)
   f.fit(X_train, y_train)

.. execute::
   :context:
   :show-source-link: yes
   :include-source: no

   fig, ax = plt.subplots()
   for _, (start, length, _) in f.embedding_.attributes:
      end = start + length
      ax.axvspan(
         start,
         end,
         ymin=0.02,
         ymax=0.98,
         facecolor="gray",
         edgecolor="black",
         alpha=0.1,
      )

   ax.plot(X_train[0])

We can also randomly sample equally sized intervals by selecting a specific
sample of those intervals, defined by ``sample_size``. This approach allows us
to focus on a subset of the intervals for detailed analysis, rather than
considering all intervals.

.. execute::
   :context:
   :show-return:

   f = IntervalTransform(
      intervals="fixed", n_intervals=30, sample_size=0.5, random_state=1
   )
   f.fit(X_train, y_train)

.. execute::
   :context:
   :show-source-link: yes
   :include-source: no

   fig, ax = plt.subplots()
   for _, (start, length, _) in f.embedding_.attributes:
      end = start + length
      ax.axvspan(
         start,
         end,
         ymin=0.02,
         ymax=0.98,
         facecolor="gray",
         edgecolor="black",
         alpha=0.1,
      )

   ax.plot(X_train[0])

Random intervals
================

Interval transformation with randomly sized intervals is a method used in time
series analysis where the data is divided into a random number of intervals,
each with a size that is randomly determined. Specifically, the number of
intervals, ``n_intervals``, is defined in advance, but the size of each
interval is sampled randomly between a minimum size (``min_size``) and a
maximum size (``max_size``), both expressed as fractions of the total size of
the input data.

This approach introduces variability into the analysis, allowing for the
exploration of patterns that might not be captured by fixed or equally sized
intervals. By varying the size of the intervals, we can potentially uncover
different trends, anomalies, or seasonal effects that may be hidden when using
more traditional, uniform interval methods.

.. execute::
   :context:

   f = IntervalTransform(
      intervals="random", n_intervals=30, min_size=0.05, max_size=0.1, random_state=1
   )
   f.fit(X_train, y_train)

.. execute::
   :context:
   :show-source-link:
   :include-source: no

   fig, ax = plt.subplots()
   for _, (start, length, _) in f.embedding_.attributes:
      end = start + length
      ax.axvspan(
         start,
         end,
         ymin=0.02,
         ymax=0.98,
         facecolor="gray",
         edgecolor="black",
         alpha=0.1,
      )

   ax.plot(X_train[0])

Dyadic intervals
================

Dyadic interval transformation is a method
where the time series data is recursively divided into smaller and smaller
intervals, with each level of recursion (or depth) producing a set of intervals
that are twice as many as the previous level. Specifically, at each depth, the
number of intervals is determined by :math:`2^{\text{depth}}`, meaning:

*	Depth 0: The entire time series is considered as a single interval.
*	Depth 1: The series is divided into 2 equal-sized intervals.
*	Depth 2: Each of the intervals from Depth 1 is further divided into 2, resulting in 4 intervals.
*	Depth 3: Each of the intervals from Depth 2 is divided again, resulting in 8 intervals.

This process continues recursively, producing increasingly smaller and more
granular intervals at each depth. Dyadic interval transformation is
particularly effective for capturing patterns at multiple scales, allowing for
a hierarchical analysis of the data. For time series classification, the method was first described by Depmster et al. (2024) [#quant]_.

.. execute::
   :context:

   f = IntervalTransform(intervals="dyadic", depth=5)
   f.fit(X_train, y_train)


.. execute::
   :context:
   :show-source-link:
   :include-source: no

   import math

   def binsearch_depth(i):

      low = 0
      high = math.ceil(math.log2(i + 2))

      while low < high:
         mid = (low + high) // 2
         if 2 ** (mid + 1) - 2 - mid > i:
            high = mid
         else:
            low = mid + 1
      return low

   fig, ax = plt.subplots(ncols=2, figsize=(12, 4))

   step = 1.0 / f.depth
   n_first = 2**f.depth - 1
   for i, (_, (start, length, _)) in enumerate(f.embedding_.attributes[:n_first]):
      end = start + length
      depth = math.floor(math.log2(i + 1))
      ax[0].axvspan(
         start,
         end,
         ymin=1 - (step * depth) - 0.02,
         ymax=1 - (step * (depth + 1)),
         facecolor="gray",
         edgecolor="black",
         alpha=0.1,
      )

   for i, (_, (start, length, _)) in enumerate(f.embedding_.attributes[n_first:]):
      end = start + length
      depth = binsearch_depth(i)
      ax[1].axvspan(
         start,
         end,
         ymin=1 - (step * depth) - 0.02,
         ymax=1 - (step * (depth + 1)),
         facecolor="gray",
         edgecolor="black",
         alpha=0.1,
      )

   ax[0].plot(X_train[0])
   ax[1].plot(X_train[0])

On the left side, we observe dyadic intervals beginning at the first time step, while on the right side, the same dyadic intervals are shifted to start in the middle of the first interval. This adjustment helps capture features in the overlapping regions between intervals.

.. _interval_summarizers:

Feature summarizers
===================
Regardless of the interval type, the :class:`~wildboar.transform.IntervalTransform` accommodates various *summarizers* to calculate one or more features per interval.

``"mean"``
  The mean of the interval. No additional parameters.

``"variance"``
  The variance of the interval. No additional parameters.

``"slope"``
  The slope of the interval. No additional parameters.

``"mean_var_slope"``
  The three values: *mean*, *variance* and *slope* for each interval. No additional parameters.

``"catch22"``
   The 22 *catch22* features. No additional parameters.

``"quant"``
  The `k = interval_length/v` quantiles of the interval. Accepts an additional
  parameter ``v``, e.g, ``summarizer_params={"v": 6}``.

A list of functions accepting a nd-array, returning a float
  The values returned by the functions

.. note::
   If the summarizer allows additional parameters, we can provide them using ``summarizer_params``
   as a ``dict`` containing parameter names and their values.


Examples
--------
Fixed intervals with four intervals and we compute the mean of each interval.

.. execute::
   :context:
   :show-return:

   f = IntervalTransform(n_intervals=4, summarizer="mean")
   f.fit_transform(X_train[:2])

Dyadic intervals with a depth of 3 we compute every fourth quantile of the intervals.

.. execute::
   :context:
   :show-return:

   f = IntervalTransform(
      intervals="dyadic", depth=3, summarizer="quant", summarizer_params={"v": 4}
   )
   f.fit_transform(X_train[:2])


**********
References
**********

.. [#rocket] Dempster, Angus, François Petitjean, and Geoffrey I. Webb. “ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels.” Data Mining and Knowledge Discovery 34, no. 5 (2020): 1454–95. https://doi.org/10.1007/s10618-020-00701-z.

.. [#hydra] Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb. “Hydra: Competing Convolutional Kernels for Fast and Accurate Time Series Classification.” Data Mining and Knowledge Discovery 37, no. 5 (2023): 1779–1805. https://doi.org/10.1007/s10618-023-00939-3.

.. [#quant] Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb. “Quant: A Minimalist Interval Method for Time Series Classification.” Data Mining and Knowledge Discovery 38, no. 4 (July 1, 2024): 2377–2402. https://doi.org/10.1007/s10618-024-01036-9.

