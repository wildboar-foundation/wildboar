from ._base import BaseFeatureEngineerTransform
from ._chydra import HydraFeatureEngineer, NormalWeightSampler


class HydraMixin:
    def _get_feature_engineer(self, n_samples):
        return HydraFeatureEngineer(8, 4, 9, NormalWeightSampler())


class HydraTransform(HydraMixin, BaseFeatureEngineerTransform):
    def __init__(self, n_jobs=None, random_state=None):
        super().__init__(n_jobs=n_jobs, random_state=random_state)
