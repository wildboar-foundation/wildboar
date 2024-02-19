import numbers
from copy import deepcopy

import numpy as np
from sklearn.base import is_classifier
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator
from ._base import DimensionSelectorMixin


class SequentialDimensionSelector(DimensionSelectorMixin, BaseEstimator):
    """
    Sequentially select a subset of dimensions.

    Sequentially select a set of dimensions by adding (forward)
    or removing (backward) dimensions to greedily form a subset.
    At each iteration, the algorithm chooses the best dimension
    to add or remove based on the cross validation score.

    Parameters
    ----------
    estimator : estimator
        An unfitted estimator.
    n_dims : {"auto"} or int, optional
        The number of dimensions to select.

        If `"auto"`, the behavior depends on `tol`:

        - if `tol` is not `None`, dimensions are selected as long as the
          increase in performance is larger than `tol`.
        - otherwise, we select half of the dimensions.

        If integer, `n_dims` is the number of dimensions to select.
    cv : int, cross-validation generator or an iterable, optional
        The cross-validation splitting strategy.
    scoring : str or callable, optional
        A str (see: :ref:`sklearn:scoring_parameter`) or callable to evaluate
        the predictions on the test set.
    direction : {"forward", "backward"}, optional
        Backward of forward selection.
    tol : float, optional
        The tolerance. If the score is not increased by `tol` between two
        iterations, return.

        If `direction="backward"`, `tol` can be negative to reduce the number
        of dimensions.

    Examples
    --------

    >>> from wildboar.datasets import load_ering
    >>> from wildboar.dimension_selection import SequentialDimensionSelector
    >>> from wildboar.distance import KNeighborsClassifier
    >>> X, y = load_ering()
    >>> clf = KNeighborsClassifier()
    >>> sds = SequentialDimensionSelector(clf, n_dims=2)
    >>> sds.fit(X, y)
    SequentialDimensionSelector(estimator=KNeighborsClassifier(), n_dims=2)
    >>> sds.get_dimensions()
    array([ True, False, False,  True])
    >>> sds.transform(X).shape
    (300, 2, 65)
    """

    _parameter_constraints = {
        "estimator": [HasMethods("fit")],
        "direction": [StrOptions({"forward", "backward"})],
        "cv": [Interval(numbers.Integral, 1, None, closed="left")],
        "scoring": [None, StrOptions(set(get_scorer_names())), callable],
        "n_dims": [
            StrOptions({"auto"}),
            Interval(numbers.Integral, 1, None, closed="left"),
        ],
        "tol": [None, Interval(numbers.Real, None, None, closed="neither")],
    }

    def __init__(
        self,
        estimator,
        *,
        n_dims="auto",
        cv=5,
        scoring=None,
        direction="forward",
        tol=None,
    ):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.n_dims = n_dims
        self.direction = direction
        self.tol = tol

    def fit(self, X, y):
        """
        Learn the dimensions to select.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep)
            The training samples.
        y : array-like of shape (n_samples, )
            The training labels.

        Returns
        -------
        object
            The instance itself.
        """
        X = self._validate_data(X, allow_3d=True, ensure_min_dims=2)
        if self.n_dims == "auto":
            if self.tol is None:
                n_dims = self.n_dims_in_ // 2
            else:
                n_dims = self.n_dims_in_
        elif (
            isinstance(self.n_dims, numbers.Integral) and self.n_dims >= self.n_dims_in_
        ):
            raise ValueError("n_dims must be < n_dims")
        else:
            n_dims = self.n_dims

        estimator = deepcopy(self.estimator)
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        mask = np.zeros(self.n_dims_in_, dtype=bool)
        current_score = -np.inf

        for i in range(n_dims):
            dim_idx, score = self._select_dimension(estimator, mask, X, y, cv)
            if self.tol is not None and score - current_score < self.tol:
                break

            current_score = score
            mask[dim_idx] = True
            if self.direction == "backward":
                mask = ~mask

        self.dimensions_ = mask
        self.n_dims_selected = mask.sum()
        return self

    def _select_dimension(self, estimator, mask, X, y, cv):
        best_score = -np.inf
        best_candidate = None
        candidates = np.flatnonzero(~mask)
        for candidate in candidates:
            new_mask = mask.copy()
            new_mask[candidate] = True

            if self.direction == "backward":
                mask = ~mask

            X_new = X[:, new_mask, :]
            if X.shape[1] == 1:
                X_new = np.squeeze(X_new, axis=1)

            score = cross_val_score(
                estimator, X_new, y, cv=cv, scoring=self.scoring
            ).mean()

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate, best_score

    def _get_dimensions(self):
        check_is_fitted(self)
        return self.dimensions_
