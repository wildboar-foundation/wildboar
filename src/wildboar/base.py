# Authors: Isak Samsten
# License: BSD 3 clause

import warnings

import numpy as np
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.utils import deprecated
from sklearn.utils.validation import _check_y, check_is_fitted

from . import __version__
from .utils.validation import _num_timesteps, check_array, check_X_y

__all__ = ["BaseEstimator"]

_DEFAULT_TAGS = {
    # The estimator can be fit with variable length time series
    "allow_eos": False,
    # The explainer requires an estimator to fit
    "requires_estimator": False,
    # X_types = ["3darray"]
}


class BaseEstimator(SklearnBaseEstimator):

    # Same additions as scikit-learn
    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        if type(self).__module__.startswith("wildboar."):
            return dict(state.items(), _wildboar_version=__version__)
        else:
            return state

    # Same check as scikit-learn
    def __setstate__(self, state):
        if type(self).__module__.startswith("wildboar."):
            pickle_version = state.pop("_wildboar_version", "pre-1.1")
            if pickle_version != __version__:
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        self.__class__.__name__, pickle_version, __version__
                    ),
                    UserWarning,
                )
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)

    def _more_tags(self):
        return _DEFAULT_TAGS

    # Disable feature names since there are no feature names
    # NOTE: Consider adding support for named dimensions for multivariate time series.
    def _check_feature_names(self, X, *, reset):
        pass

    def _check_n_features(self, X, reset):
        self._check_n_timesteps(X, reset)

    def _check_n_timesteps(self, X, reset):
        try:
            n_timesteps, n_dims = _num_timesteps(X)
        except TypeError as e:
            if not reset and hasattr(self, "n_timesteps_in_"):
                raise ValueError(
                    "X does not contain any timesteps, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_timesteps_in_} timesteps"
                ) from e
            # If the number of timesteps is not defined and reset=True,
            # then we skip this check
            return

        if reset:
            self.n_timesteps_in_ = n_timesteps
            self.n_dims_in_ = n_dims

            # Set n_features_in_ for compatibility with scikit-learn
            self.n_features_in_ = n_timesteps * n_dims
            return

        if not hasattr(self, "n_timesteps_in_") and not hasattr(self, "n_dims_in_"):
            # Skip this check if the expected number of expected input features
            # was not recorded by calling fit first. This is typically the case
            # for stateless transformers.
            return

        if n_timesteps != self.n_timesteps_in_:
            raise ValueError(
                f"X has {n_timesteps} timesteps, but {self.__class__.__name__} "
                f"is expecting {self.n_timesteps_in_} timesteps as input."
            )
        if n_dims != self.n_dims_in_:
            raise ValueError(
                f"X has {n_dims} dimensions, but {self.__class__.__name__} "
                f"is expecting {self.n_dims_in_} dimensions as input."
            )

    def _validate_force_n_dims(self, X):
        """Validate that X is valid in relation to the optional _force_n_dims attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, m)
            The array to validate.

        Returns
        -------
        X : ndarray of shape (n_samples, self._force_n_dims, -1)
            The array validated and reshaped
        """
        if hasattr(self, "_force_n_dims"):
            X = np.asarray(X)
            if X.ndim == 2:
                return X.reshape(X.shape[0], self._force_n_dims, -1)
            elif X.ndim == 3 and X.shape[1] != self._force_n_dims:
                raise ValueError(
                    f"{self.__class__.__name__} has _force_n_dims set to "
                    f"{self._force_n_dims}, but X has {X.shape[1]} dimensions as input."
                )

        return X

    # We override sklearn but delegate to wildboar's check_array
    # and check_X_y
    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        **check_params,
    ):
        if y is None and self._get_tags()["requires_y"]:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )

        no_val_X = isinstance(X, str) and X == "no_validation"
        no_val_y = y is None or isinstance(y, str) and y == "no_validation"

        default_check_params = {"estimator": self}
        check_params = {**default_check_params, **check_params}

        if not no_val_X:
            X = self._validate_force_n_dims(X)

        if no_val_X and no_val_y:
            raise ValueError("Validation should be done on X, y or both.")
        elif not no_val_X and no_val_y:
            X = check_array(X, input_name="X", **check_params)
            out = X
        elif no_val_X and not no_val_y:
            y = _check_y(y, **check_params)
            out = y
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                if "estimator" not in check_X_params:
                    check_X_params = {**default_check_params, **check_X_params}
                X = check_array(X, input_name="X", **check_X_params)
                if "estimator" not in check_y_params:
                    check_y_params = {**default_check_params, **check_y_params}
                y = check_array(y, input_name="y", **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if not no_val_X:
            self._check_n_timesteps(X, reset=reset)

        return out


class ExplainerMixin:
    """Mixin class for all explainers in wildboar"""

    _estimator_type = "explainer"

    def _validate_estimator(self, estimator, allow_3d=False):
        """Check the estimator and set the `n_timesteps_in_` and `n_dims_in_`
        parameter from the estimator.

        Parameters
        ----------
        estimator : object
            The estimator object to check

        allow_3d : bool, optional
            If estimator fit with 3d-arrays are supported.
        """
        check_is_fitted(estimator)
        if hasattr(estimator, "n_timesteps_in_"):
            self.n_timesteps_in_ = estimator.n_timesteps_in_
        elif hasattr(estimator, "n_features_in_"):
            self.n_timesteps_in_ = estimator.n_features_in_
        else:
            raise ValueError(
                "Unable to find the number of timesteps from %s. Please ensure that "
                "the estimator is correctly fit and sets the n_timesteps_in_ or "
                "n_features_in_ properties." % type(estimator).__qualname__
            )

        self.n_features_in_ = self.n_timesteps_in_
        if hasattr(estimator, "n_dims_in_"):
            self.n_dims_in_ = estimator.n_dims_in_
        else:
            self.n_dims_in_ = 1

        if self.n_dims_in_ > 1 and not allow_3d:
            raise ValueError(
                "The explainer does not permit 3darrays as input but the estimator is "
                "fitted with a 3darray of shape (?, %d, %d)."
                % (self.n_dims_in_, self.n_timesteps_in_)
            )

        return estimator

    def fit_explain(self, estimator, x=None, y=None, **kwargs):
        return self.fit(estimator, x, y, **kwargs).explain(x, y)

    def plot(self, x=None, y=None, ax=None):
        """Plot the explanation.

        Returns
        -------
        ax : Axes
            The axes object
        """
        from .utils.plot import plot_time_domain

        plot_time_domain(self.explain(x, y), y, ax=ax)

    def _more_tags(self):
        return {"requires_estimator": True}


class CounterfactualMixin:
    """Mixin class for counterfactual explainer."""

    _estimator_type = "counterfactual"

    @deprecated(
        "transform(x, y) has been deprecated in 1.1 and will be removed in 1.2. "
        "Use explain(x, y) instead."
    )
    def transform(self, x, y):
        return self.explain(x, y)

    def score(self, x, y):
        """Score the counterfactual explainer in terms of closeness of fit.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep)
            The samples.

        y : array-like of shape (n_samples, )
            The desired counterfactal label.

        Returns
        -------
        score : float
            The closensess of fit.
        """
        from .metrics import proximity_score

        return proximity_score(x, self.explain(x, y))


def is_counterfactual(estimator):
    """Check if estimator is a counterfactual explainer

    Parameters
    ----------
    estimator : object
        The estimator

    Returns
    -------
    bool
        True if the estimator probably is a counterfactual explainer
    """
    return (
        hasattr(estimator, "explain")
        and hasattr(type(estimator), "_estimator_type")
        and type(estimator)._estimator_type == "counterfacutal"
    )


def is_explainer(estimator):
    """Check if estimator is an explainer

    Parameters
    ----------
    estimator : object
        The estimator

    Returns
    -------
    bool
        True if the estimator probably is an explainer"""
    return hasattr(estimator, "explain")
