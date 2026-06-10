"""Fabric: molecular ML orchestration on Grumpy.

Fabric wires datasets, benchmarks, models, and workflows into a single
config-driven stack for biomolecular machine learning. Use :class:`~fabric.utils.settings.Settings`
to scope storage paths and :mod:`fabric.core.factory` to construct objects from
YAML configs.

Example:
    >>> from fabric import Settings
    >>> Settings.DEFAULT_HOME
    PosixPath('~/.imaginary')
"""

from fabric._version import __version__
from fabric.utils.settings import Settings

__all__ = ["Settings", "__version__"]
