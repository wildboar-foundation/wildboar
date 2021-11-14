from abc import abstractmethod
from collections import namedtuple

from wildboar.explain.base import BaseExplanation

Importance = namedtuple("Importance", ["mean", "std", "full"])


class BaseImportance(BaseExplanation):
    def __init__(self, *, scoring=None):
        self.scoring = scoring

    @abstractmethod
    def fit(self, x, y=None, sample_weight=None):
        pass

    @abstractmethod
    def score(self, timestep=None, dim=0):
        pass
