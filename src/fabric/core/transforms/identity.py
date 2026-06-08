from __future__ import annotations

from fabric.core.transform import Transform


class Identity(Transform):
    """No-op transform that returns inputs unchanged.

    Used as the default when a dataset config omits ``transforms``.

    Example:
        >>> from fabric.core.transforms import Identity
        >>> step = Identity()
        >>> list(step.transform_batches(iter([batch])))
    """

    name = "Identity"
