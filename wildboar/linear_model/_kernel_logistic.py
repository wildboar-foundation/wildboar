from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class KernelLogisticRegression(LogisticRegression):
    """A simple kernel logistic implementation using a Nystroem kernel approximation

    Warnings
    --------
    This kernel method is not specialized for temporal classification.

    See Also
    --------
    wildboar.datasets.outlier.EmmottLabeler : Synthetic outlier dataset construction
    """

    def __init__(self, kernel=None, *, kernel_params=None, n_components=100, **kwargs):
        """Create a new kernel logistic regression

        Parameters
        ----------
        kernel : str, optional
            The kernel function to use. See `sklearn.metrics.pairwise.kernel_metric` for kernels. The default kernel
            is 'rbf'.

        kernel_params : dict, optional
            Parameters to the kernel function.

        n_components : int, optional
            Number of features to construct

        kwargs : dict, optional
            Parameters to the logistic regression model. See `sklearn.linear_model.LogisticRegression`.
        """
        super().__init__(**kwargs)
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.n_components = n_components

    def fit(self, x, y, sample_weight=None):
        self.nystroem_ = Nystroem(kernel=self.kernel or 'rbf', kernel_params=self.kernel_params,
                                  n_components=self.n_components)
        self.nystroem_.fit(x)
        super().fit(self.nystroem_.transform(x), y, sample_weight=sample_weight)
        return self

    def decision_function(self, x):
        check_is_fitted(self)
        return super().decision_function(self.nystroem_.transform(x))
