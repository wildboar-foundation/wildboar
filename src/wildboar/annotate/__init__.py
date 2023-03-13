# Authors: Isak Samsten
# License: BSD 3 clause

"""Annotation of time series.

See the :doc:`annotation section in the User Guide
<wildboar:guide/annotate>` for more details and examples.
"""
from ._motifs import motifs
from ._segment import segment

__all__ = ["motifs", "segment"]
