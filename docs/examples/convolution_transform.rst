#####################
Convolution transform
#####################

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


********************
Hydra transformation
********************

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

****************
Rocket transform
****************

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

.. [#rocket] Rocket

.. [#hydra] Hydra
