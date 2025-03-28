from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass

from sklearn.utils._tags import Tags, _dataclass_args


@dataclass(**_dataclass_args())
class ExplainerTags:
    require_estimator: bool = True


@dataclass(**_dataclass_args())
class WildboarTags(Tags):
    allow_eos: bool = False
    explainer_tags: ExplainerTags | None = None


def _tags_from_sklearn(tags: Tags) -> WildboarTags:
    if sys.version_info >= (3, 10):
        sklearn_tags = {slot: getattr(tags, slot) for slot in Tags.__slots__}
    else:
        sklearn_tags = {
            field.name: getattr(tags, field.name) for field in dataclasses.fields(Tags)
        }
    return WildboarTags(explainer_tags=None, **sklearn_tags)
