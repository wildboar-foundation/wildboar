import numbers

from sklearn.utils._param_validation import Interval, StrOptions

from ._base import BaseAttributeTransform
from ._chydra import HydraAttributeGenerator, NormalKernelSampler

_SAMPLING_METHOD = {
    "normal": NormalKernelSampler,
}


class HydraMixin:
    _parameter_constraints: dict = {
        "n_kernels": [Interval(numbers.Integral, 1, None, closed="left")],
        "kernel_size": [Interval(numbers.Integral, 2, None, closed="left")],
        "n_groups": [Interval(numbers.Integral, 1, None, closed="left")],
        "sampling": [StrOptions({"normal"})],
        "sampling_params": [dict, None],
    }

    def _get_generator(self, x, y):
        sampling_params = {} if self.sampling_params is None else self.sampling_params
        return HydraAttributeGenerator(
            self.n_groups,
            self.n_kernels,
            self.kernel_size,
            _SAMPLING_METHOD[self.sampling](**sampling_params),
        )


class HydraTransform(HydraMixin, BaseAttributeTransform):
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
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means using
        a single core and a value of `-1` means using all cores. Positive
        integers mean the exact number of cores.
    random_state : int or RandomState, optional
        Controls the random sampling of kernels.

        - If `int`, `random_state` is the seed used by the random number
          generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.

    Attributes
    ----------
    embedding_ : Embedding
        The underlying embedding

    See Also
    --------
    HydraClassifier : A classifier using hydra transform.

    Notes
    -----
    The implementation does not implement the first order descrete differences
    described by Dempster et. al. (2023). If this is desired, one can use
    native scikit-learn functionalities and the
    :class:`~wildboar.transform.DiffTransform`:

    >>> from sklearn.pipeline import make_pipeline, make_union
    >>> from wildboar.transform import DiffTransform, HydraTransform
    >>> dempster_hydra = make_union(
    ...     HydraTransform(n_groups=32),
    ...     make_pipeline(
    ...         DiffTransform(),
    ...         HydraTransform(n_groups=32)
    ...     )
    ... )

    References
    ----------
    Dempster, A., Schmidt, D. F., & Webb, G. I. (2023).
        Hydra: competing convolutional kernels for fast and accurate
        time series classification. Data Mining and Knowledge Discovery

    Examples
    --------
    >>> from wildboar.datasets import load_gun_point
    >>> from wildboar.transform import HydraTransform
    >>> X, y = load_gun_point()
    >>> t = HydraTransform(n_groups=8, n_kernels=4, random_state=1)
    >>> t.fit_transform(X)
    """

    _parameter_constraints: dict = {
        **HydraMixin._parameter_constraints,
        **BaseAttributeTransform._parameter_constraints,
    }

    def __init__(
        self,
        *,
        n_groups=64,
        n_kernels=8,
        kernel_size=9,
        sampling="normal",
        sampling_params=None,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.n_groups = n_groups
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.sampling = sampling
        self.sampling_params = sampling_params
