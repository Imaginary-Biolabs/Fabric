"""Collater registry."""

from fabric.core.collaters.long import LongCollater
from fabric.core.collaters.wide import WideCollater

__all__ = ["LongCollater", "WideCollater"]
