# Authors: Isak Samsten
# License: BSD 3 clause

from ..transform import RocketTransform
from ._transform import TransformRidgeClassifierCV, TransformRidgeCV


class RocketClassifier(TransformRidgeClassifierCV):
    """
    A classifier using Rocket transform.

    Parameters
    ----------
    n_kernels : int, optional
        The number of kernels to sample at each node.
    sampling : {"normal", "uniform", "shapelet"}, optional
        The sampling of convolutional filters.

        - if "normal", sample filter according to a normal distribution with
          ``mean`` and ``scale``.
        - if "uniform", sample filter according to a uniform distribution with
          ``lower`` and ``upper``.
        - if "shapelet", sample filters as subsequences in the training data.
    sampling_params : dict, optional
        Parameters for the sampling strategy.

        - if "normal", ``{"mean": float, "scale": float}``, defaults to
          ``{"mean": 0, "scale": 1}``.
        - if "uniform", ``{"lower": float, "upper": float}``, defaults to
          ``{"lower": -1, "upper": 1}``.
    kernel_size : array-like, optional
        The kernel size, by default ``[7, 11, 13]``.
    min_size : float, optional
        The minimum timestep size used for generating kernel sizes, If set,
        ``kernel_size`` is ignored.
    max_size : float, optional
        The maximum timestep size used for generating kernel sizes, If set,
        ``kernel_size`` is ignored.
    bias_prob : float, optional
        The probability of using the bias term.
    normalize_prob : float, optional
        The probability of performing normalization.
    padding_prob : float, optional
        The probability of padding with zeros.
    alphas : array-like of shape (n_alphas,), optional
        Array of alpha values to try.
    fit_intercept : bool, optional
        Whether to calculate the intercept for this model.
    scoring : str, callable, optional
        A string or a scorer callable object with signature
        `scorer(estimator, X, y)`.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form `{class_label: weight}`.
    normalize : "sparse" or bool, optional
        Standardize before fitting. By default use
        :class:`datasets.preprocess.SparseScaler` to standardize the attributes. Set
        to `False` to disable or `True` to use `StandardScaler`.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If ``int``, ``random_state`` is the seed used by the random number
          generator.
        - If :class:`numpy.random.RandomState` instance, ``random_state`` is
          the random number generator.
        - If ``None``, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of ``None`` means using
        a single core and a value of ``-1`` means using all cores. Positive
        integers mean the exact number of cores.
    """

    _parameter_constraints = {
        **TransformRidgeClassifierCV._parameter_constraints,
        **RocketTransform._parameter_constraints,
    }

    def __init__(  # noqa: PLR0913
        self,
        n_kernels=10000,
        *,
        sampling="normal",
        sampling_params=None,
        kernel_size=None,
        min_size=None,
        max_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        scoring=None,
        cv=None,
        class_weight=None,
        normalize=True,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            scoring=scoring,
            cv=cv,
            class_weight=class_weight,
            n_jobs=n_jobs,
            normalize=normalize,
            random_state=random_state,
        )
        self.padding_prob = padding_prob
        self.normalize_prob = normalize_prob
        self.bias_prob = bias_prob
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.min_size = min_size
        self.max_size = max_size

    def _get_transform(self, random_state):
        return RocketTransform(
            self.n_kernels,
            kernel_size=self.kernel_size,
            max_size=self.max_size,
            min_size=self.max_size,
            sampling=self.sampling,
            sampling_params=self.sampling_params,
            bias_prob=self.bias_prob,
            normalize_prob=self.normalize_prob,
            padding_prob=self.padding_prob,
            random_state=random_state,
            n_jobs=self.n_jobs,
        )


class RocketRegressor(TransformRidgeCV):
    """
    A regressor using Rocket transform.

    Parameters
    ----------
    n_kernels : int, optional
        The number of kernels to sample at each node.
    sampling : {"normal", "uniform", "shapelet"}, optional
        The sampling of convolutional filters.

        - if "normal", sample filter according to a normal distribution with
          ``mean`` and ``scale``.
        - if "uniform", sample filter according to a uniform distribution with
          ``lower`` and ``upper``.
        - if "shapelet", sample filters as subsequences in the training data.
    sampling_params : dict, optional
        Parameters for the sampling strategy.

        - if "normal", ``{"mean": float, "scale": float}``, defaults to
          ``{"mean": 0, "scale": 1}``.
        - if "uniform", ``{"lower": float, "upper": float}``, defaults to
          ``{"lower": -1, "upper": 1}``.
    kernel_size : array-like, optional
        The kernel size, by default ``[7, 11, 13]``.
    min_size : float, optional
        The minimum timestep size used for generating kernel sizes, If set,
        ``kernel_size`` is ignored.
    max_size : float, optional
        The maximum timestep size used for generating kernel sizes, If set,
        ``kernel_size`` is ignored.
    bias_prob : float, optional
        The probability of using the bias term.
    normalize_prob : float, optional
        The probability of performing normalization.
    padding_prob : float, optional
        The probability of padding with zeros.
    alphas : array-like of shape (n_alphas,), optional
        Array of alpha values to try.
    fit_intercept : bool, optional
        Whether to calculate the intercept for this model.
    scoring : str, callable, optional
        A string or a scorer callable object with signature
        `scorer(estimator, X, y)`.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    gcv_mode : {'auto', 'svd', 'eigen'}, optional
        Flag indicating which strategy to use when performing
        Leave-One-Out Cross-Validation. Options are::

            'auto' : use 'svd' if n_samples > n_features, otherwise use 'eigen'
            'svd' : force use of singular value decomposition of X when X is
                dense, eigenvalue decomposition of X^T.X when X is sparse.
            'eigen' : force computation via eigendecomposition of X.X^T

        The 'auto' mode is the default and is intended to pick the cheaper
        option of the two depending on the shape of the training data.
    normalize : "sparse" or bool, optional
        Standardize before fitting. By default use
        :class:`datasets.preprocess.SparseScaler` to standardize the attributes. Set
        to `False` to disable or `True` to use `StandardScaler`.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If ``int``, ``random_state`` is the seed used by the random number
          generator.
        - If :class:`numpy.random.RandomState` instance, ``random_state`` is
          the random number generator.
        - If ``None``, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of ``None`` means using
        a single core and a value of ``-1`` means using all cores. Positive
        integers mean the exact number of cores.
    """

    _parameter_constraints = {
        **TransformRidgeCV._parameter_constraints,
        **RocketTransform._parameter_constraints,
    }

    def __init__(  # noqa: PLR0913
        self,
        n_kernels=10000,
        *,
        sampling="normal",
        sampling_params=None,
        kernel_size=None,
        min_size=None,
        max_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        scoring=None,
        cv=None,
        gcv_mode=None,
        normalize=True,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            scoring=scoring,
            cv=cv,
            gcv_mode=gcv_mode,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.padding_prob = padding_prob
        self.normalize_prob = normalize_prob
        self.bias_prob = bias_prob
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.min_size = min_size
        self.max_size = max_size

    def _get_transform(self, random_state):
        return RocketTransform(
            self.n_kernels,
            kernel_size=self.kernel_size,
            max_size=self.max_size,
            min_size=self.max_size,
            sampling=self.sampling,
            sampling_params=self.sampling_params,
            bias_prob=self.bias_prob,
            normalize_prob=self.normalize_prob,
            padding_prob=self.padding_prob,
            random_state=random_state,
            n_jobs=self.n_jobs,
        )
