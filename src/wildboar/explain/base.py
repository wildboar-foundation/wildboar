import abc

from sklearn.base import BaseEstimator

__all__ = [
    "BaseExplanation",
]


class BaseExplanation(BaseEstimator, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def plot(ax=None, **kwargs):
        pass
