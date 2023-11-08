# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_random_state

from ..datasets.preprocess import SparseScaler
from ..transform import DiffTransform, HydraTransform
from ._transform import TransformRidgeClassifierCV


class HydraClassifier(TransformRidgeClassifierCV):
    """
    A Dictionary based method using convolutional kernels.

    Parameters
    ----------
    n_groups : int, optional
        The number of groups of kernels.
    n_kernels : int, optional
        The number of kernels per group.
    kernel_size : int, optional
        The size of the kernel.
    sampling : {"normal"}, optional
        The strategy for sampling kernels. By default kernel weights
        are sampled from a normal distribution with zero mean and unit
        standard deviation.
    sampling_params : dict, optional
        Parameters to the sampling approach. The "normal" sampler
        accepts two parameters: `mean` and `scale`.
    order : int, optional
        The order of difference. If set, half the groups with corresponding
        kernels will be convolved with the `order` discrete difference along
        the time dimension.
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
    normalize : bool, optional
        Standardize before fitting. By default use
        :class:`datasets.preprocess.SparseScaler` to standardize the attributes. Set
        to `False` to disable or `True` to use `StandardScaler`.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means using
        a single core and a value of `-1` means using all cores. Positive
        integers mean the exact number of cores.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the random number
          generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.

    References
    ----------
    Dempster, A., Schmidt, D. F., & Webb, G. I. (2023).
        Hydra: competing convolutional kernels for fast and accurate
        time series classification. Data Mining and Knowledge Discovery
    """

    def __init__(
        self,
        *,
        n_groups=64,
        n_kernels=8,
        kernel_size=9,
        sampling="normal",
        sampling_params=None,
        order=1,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        scoring=None,
        cv=None,
        class_weight=None,
        normalize="sparse",
        n_jobs=None,
        random_state=None,
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
        self.n_groups = n_groups
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.order = order

    def _build_pipeline(self):
        pipeline = super()._build_pipeline()
        if self.normalize == "sparse":
            pipeline[1] = ("normalize", SparseScaler())

        return pipeline

    def _get_transform(self, random_state):
        if self.order is not None and self.order > 0 and self.n_groups > 1:
            random_state = check_random_state(self.random_state)
            return make_union(
                HydraTransform(
                    n_groups=self.n_groups // 2,
                    n_kernels=self.n_kernels,
                    kernel_size=self.kernel_size,
                    sampling=self.sampling,
                    sampling_params=self.sampling_params,
                    random_state=random_state.randint(np.iinfo(np.int32).max),
                    n_jobs=self.n_jobs,
                ),
                make_pipeline(
                    DiffTransform(order=self.order),
                    HydraTransform(
                        n_groups=self.n_groups // 2,
                        n_kernels=self.n_kernels,
                        kernel_size=self.kernel_size,
                        sampling=self.sampling,
                        sampling_params=self.sampling_params,
                        random_state=random_state.randint(np.iinfo(np.int32).max),
                        n_jobs=self.n_jobs,
                    ),
                ),
            )

        else:
            return HydraTransform(
                n_groups=self.n_groups,
                n_kernels=self.n_kernels,
                kernel_size=self.kernel_size,
                sampling=self.sampling,
                sampling_params=self.sampling_params,
                random_state=random_state,
                n_jobs=self.n_jobs,
            )
