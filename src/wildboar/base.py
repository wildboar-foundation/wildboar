# Authors: Isak Samsten
# License: BSD 3 clause
"""Base classes for all estimators."""

import itertools
import warnings

import numpy as np
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.utils._tags import get_tags
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import _check_y, check_is_fitted

from . import __version__
from ._tags import WildboarTags, _tags_from_sklearn
from .utils.validation import _num_timesteps, check_array, check_X_y

__all__ = [
    "BaseEstimator",
    "CounterfactualMixin",
    "ExplainerMixin",
    "is_counterfactual",
    "is_explainer",
]

_DEFAULT_TAGS = {
    # The estimator can be fit with variable length time series
    "allow_eos": False,
    # By default no estimator supports nan.
    "allow_nan": False,
    # The explainer requires an estimator to fit
    "requires_estimator": False,
    # X_types = ["3darray"]
}


class BaseEstimator(SklearnBaseEstimator):
    """Base estimator for all Wildboar estimators."""

    _doc_link_module = "wildboar"

    @property
    def _doc_link_template(self):
        """Get the documentation link template for the current wildboar version.

        The template generates documentation URLs that point to either a specific version
        or the master branch depending on whether this is a development version.

        Returns
        -------
        str
            A URL template string containing placeholders for {module_path},
            {estimator_module}, and {estimator_name}.

        Notes
        -----
        For released versions, links point to that specific version (e.g., "1.2").
        For development versions, links point to the "master" branch.
        """
        wildboar_version = parse_version(__version__)
        if wildboar_version.dev is None:
            version_url = f"{wildboar_version.major}.{wildboar_version.minor}"
        else:
            version_url = "master"
        return (
            "https://wildboar.dev/%s/api/{module_path}/index.html#{estimator_module}.{estimator_name}"
            % version_url
        )

    def _doc_link_url_param_generator(self, other):
        """
        Generate URL parameter dictionary for documentation linking.

        Parameters
        ----------
        other : object
            Another estimator object used for comparison.

        Returns
        -------
        dict
            Dictionary containing URL components with keys:
            - 'module_path': Module path with '/' as separator.
            - 'estimator_module': Full module path with '.' as separator.
            - 'estimator_name': Name of the estimator class.
        """
        estimator_name = self.__class__.__name__
        estimator_module = ".".join(
            itertools.takewhile(
                lambda part: not part.startswith("_"),
                self.__class__.__module__.split("."),
            )
        )
        return {
            "module_path": estimator_module.replace(".", "/"),
            "estimator_module": estimator_module,
            "estimator_name": estimator_name,
        }

    # Same additions as scikit-learn
    def __getstate__(self):
        """
        Get the state of the estimator.

        Add a new element to the dict we return `_wildboar_version` which
        is we use to warn when setting the state.

        Returns
        -------
        dict
            The state.
        """
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
        """
        Set the state of the estimator.

        Gives a warning if a user tries to unpickle a object serialized
        from a older version of Wildboar.

        Parameters
        ----------
        state : object
            The state.
        """
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

    def __sklearn_tags__(self) -> WildboarTags:
        tags = _tags_from_sklearn(super().__sklearn_tags__())
        tags.allow_eos = False
        return tags

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
                f"X has {n_timesteps} features, but {self.__class__.__name__} "
                f"is expecting {self.n_timesteps_in_} features as input."
            )
        if n_dims != self.n_dims_in_:
            raise ValueError(
                f"X has {n_dims} dimensions, but {self.__class__.__name__} "
                f"is expecting {self.n_dims_in_} dimensions as input."
            )

    def _validate_force_n_dims(self, X):
        """
        Validate that X is valid.

        The validation takes into consideration the optional _force_n_dims
        attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, m)
            The array to validate.

        Returns
        -------
        ndarray of shape (n_samples, self._force_n_dims, -1)
            The array validated and reshaped.
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

    # sklearn has recently migrated away from this, to a specific
    # method for validation. We keep it because it's easier to extend
    # for consumers of our classes.
    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        **check_params,
    ):
        if y is None and get_tags(self).target_tags.required:
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
    """Mixin class for all explainers in wildboar."""

    _estimator_type = "explainer"

    def _validate_estimator(self, estimator, allow_3d=False, require_classes=True):
        """
        Check the estimator.

        Set the `n_timesteps_in_` and `n_dims_in_` parameter from the estimator.

        Parameters
        ----------
        estimator : object
            The estimator object to check.
        allow_3d : bool, optional
            If estimator fit with 3d-arrays are supported.
        require_classes : bool, optional
            If the estimator require the `classes_` attribute.

        Returns
        -------
        estimator
            The validated estimator.
        """
        check_is_fitted(estimator)
        if hasattr(estimator, "n_timesteps_in_"):
            self.n_timesteps_in_ = estimator.n_timesteps_in_
        elif hasattr(estimator, "n_features_in_"):
            self.n_timesteps_in_ = estimator.n_features_in_
        else:
            raise ValueError(
                "Unable to find the number of timesteps from {}. Please ensure that "
                "the estimator is correctly fit and sets the n_timesteps_in_ or "
                "n_features_in_ properties.".format(type(estimator).__qualname__)
            )

        self.n_features_in_ = self.n_timesteps_in_
        if hasattr(estimator, "n_dims_in_"):
            self.n_dims_in_ = estimator.n_dims_in_
        else:
            self.n_dims_in_ = 1

        if self.n_dims_in_ > 1 and not allow_3d:
            raise ValueError(
                "The explainer does not permit 3darrays as input but the estimator is "
                "fitted with a 3darray of shape (?, {}, {}).".format(
                    self.n_dims_in_, self.n_timesteps_in_
                )
            )

        if require_classes:
            if hasattr(estimator, "classes_"):
                self.classes_ = estimator.classes_
            else:
                raise ValueError(
                    "Unable to find the `classes_` attribute from {}".format(
                        type(estimator).__qualname__
                    )
                )

        return estimator

    def fit_explain(self, estimator, x=None, y=None, **kwargs):
        """
        Fit and return the explanation.

        Parameters
        ----------
        estimator : Estimator
            The estimator to explain.
        x : time-series, optional
            The input time series.
        y : array-like of shape (n_samples, ), optional
            The labels.
        **kwargs : dict, optional
            Optional extra arguments.

        Returns
        -------
        ndarray
            The explanation.
        """
        return self.fit(estimator, x, y, **kwargs).explain(x, y)

    def plot(self, x=None, y=None, ax=None):
        """
        Plot the explanation.

        Parameters
        ----------
        x : array-like, optional
            Optional imput samples.
        y : array-like, optional
            Optional target labels.
        ax : Axes, optional
            Optional axes to plot to.

        Returns
        -------
        Axes
            The axes object.
        """
        from .utils.plot import plot_time_domain

        plot_time_domain(self.explain(x, y), y, ax=ax)

    def __sklearn_tags__(self):
        from ._tags import ExplainerTags

        tags = super().__sklearn_tags__()
        tags.estimator_type = "explainer"
        tags.explainer_tags = ExplainerTags()
        return tags


class CounterfactualMixin:
    """Mixin class for counterfactual explainer."""

    # TODO(1.4) Remove
    _estimator_type = "counterfactual"

    def score(self, x, y):
        """
        Score the counterfactual explainer in terms of closeness of fit.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep)
            The samples.
        y : array-like of shape (n_samples, )
            The desired counterfactal label.

        Returns
        -------
        float
            The proximity.
        """
        from .metrics import proximity_score

        return proximity_score(x, self.explain(x, y))

    def __sklearn_tags__(self):
        from ._tags import ExplainerTags

        tags = super().__sklearn_tags__()
        tags.estimator_type = "counterfactual"
        tags.explainer_tags = ExplainerTags()
        return tags


def is_counterfactual(estimator):
    """
    Check if estimator is a counterfactual explainer.

    Parameters
    ----------
    estimator : object
        The estimator.

    Returns
    -------
    bool
        True if the estimator probably is a counterfactual explainer.
    """
    return (
        hasattr(estimator, "explain")
        and hasattr(type(estimator), "_estimator_type")
        and type(estimator)._estimator_type == "counterfactual"
    )


def is_explainer(estimator):
    """
    Check if estimator is an explainer.

    Parameters
    ----------
    estimator : object
        The estimator.

    Returns
    -------
    bool
        True if the estimator probably is an explainer.
    """
    return hasattr(estimator, "explain")
