import warnings

from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.utils.validation import _check_y

from . import __version__
from .utils.validation import _num_timesteps, check_array, check_X_y

__all__ = ["BaseEstimator"]

_DEFAULT_TAGS = {"allow_multivariate": False, "allow_eos": False}


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
            self.n_features_in_ = n_timesteps
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

        if not no_val_X and check_params.get("ensure_2d", True):
            self._check_n_features(X, reset=reset)

        return out
