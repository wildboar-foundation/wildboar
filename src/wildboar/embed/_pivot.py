from wildboar.distance import _DISTANCE_MEASURE

from ._cpivot import PivotFeatureEngineer
from .base import BaseEmbedding


class PivotEmbedding(BaseEmbedding):
    def __init__(self, n_pivot=1, *, metrics="all", random_state=None, n_jobs=None):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.n_pivot = n_pivot
        self.metrics = metrics

    def _more_tags(self):
        return {"require_y": True}

    def _get_feature_engineer(self):
        return PivotFeatureEngineer(
            int(self.n_pivot), [_DISTANCE_MEASURE["euclidean"]()]
        )
