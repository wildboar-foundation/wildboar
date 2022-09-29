# Authors: Isak Samsten
# License: BSD 3 clause

from ..transform import RocketTransform
from ._transform import TransformRidgeClassifierCV, TransformRidgeCV


class RocketClassifier(TransformRidgeClassifierCV):
    """Implements the ROCKET classifier"""

    def __init__(
        self,
        n_kernels=10000,
        *,
        kernel_size=None,
        sampling="normal",
        sampling_params=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        normalize="deprecated",
        scoring=None,
        cv=None,
        class_weight=None,
        n_jobs=None,
        random_state=None
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            normalize=normalize,
            scoring=scoring,
            cv=cv,
            class_weight=class_weight,
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

    def _get_transform(self, random_state):
        return RocketTransform(
            self.n_kernels,
            kernel_size=self.kernel_size,
            sampling=self.sampling,
            sampling_params=self.sampling_params,
            bias_prob=self.bias_prob,
            normalize_prob=self.normalize_prob,
            padding_prob=self.padding_prob,
            random_state=random_state,
            n_jobs=self.n_jobs,
        )


class RocketRegressor(TransformRidgeCV):
    """Implements the ROCKET regressor"""

    def __init__(
        self,
        n_kernels=10000,
        *,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        normalize="deprecated",
        scoring=None,
        cv=None,
        gcv_mode=None,
        n_jobs=None,
        random_state=None
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            normalize=normalize,
            scoring=scoring,
            cv=cv,
            gcv_mode=gcv_mode,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.n_kernels = n_kernels

    def _get_transform(self, random_state):
        return RocketTransform(
            self.n_kernels, random_state=random_state, n_jobs=self.n_jobs
        )
