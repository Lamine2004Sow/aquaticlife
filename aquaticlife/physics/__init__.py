"""Composants physiques : segments, muscles, fluide et intégrateurs."""

from .body import RigidSegmentChain, MorphologyParameters
from .fluid import FluidModel

__all__ = ["RigidSegmentChain", "MorphologyParameters", "FluidModel"]
