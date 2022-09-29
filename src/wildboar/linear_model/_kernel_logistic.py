# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted, check_random_state


class KernelLogisticRegression(LogisticRegression):
    """A simple kernel logistic implementation using a Nystroem kernel approximation

    Warnings
    --------
    This kernel method is not specialized for temporal classification.

    See Also
    --------
    wildboar.datasets.outlier.EmmottLabeler : Synthetic outlier dataset construction

    """

    def __init__(
        self,
        kernel=None,
        *,
        kernel_params=None,
        n_components=100,
        penalty="l2",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None
    ):
        """
        Parameters
        ----------
        kernel : str, optional
            The kernel function to use. See `sklearn.metrics.pairwise.kernel_metric`
            for kernels. The default kernel is 'rbf'.

        kernel_params : dict, optional
            Parameters to the kernel function.

        n_components : int, optional
            Number of features to construct
        """
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.n_components = n_components

    def fit(self, x, y, sample_weight=None):
        random_state = check_random_state(self.random_state)
        kernel = self.kernel or "rbf"
        n_components = min(x.shape[0], self.n_components)
        self.nystroem_ = Nystroem(
            kernel=kernel,
            kernel_params=self.kernel_params,
            n_components=n_components,
            random_state=random_state.randint(np.iinfo(np.int32).max),
        )
        self.nystroem_.fit(x)
        super().fit(self.nystroem_.transform(x), y, sample_weight=sample_weight)
        return self

    def decision_function(self, x):
        check_is_fitted(self)
        return super().decision_function(self.nystroem_.transform(x))
