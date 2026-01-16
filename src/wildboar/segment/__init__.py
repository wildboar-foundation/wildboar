"""
Segment time series into regions.
"""

from ._base import detect_changepoints
from ._mp import FlussSegmenter

__all__ = ["FlussSegmenter", "detect_changepoints"]
