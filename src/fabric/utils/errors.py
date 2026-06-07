"""Fabric exception types."""


class FabricError(Exception):
    """Base class for Fabric errors."""


class ConfigurationError(FabricError):
    """Invalid or missing configuration."""


class SchemaError(FabricError):
    """Biological schema level missing or invalid."""


class TransformError(FabricError):
    """Transform pipeline error."""


class PlatformExtraRequired(FabricError):
    """Optional platform dependencies are not installed."""

    def __init__(self) -> None:
        super().__init__(
            "Platform features require the platform extra: pip install 'imaginary-fabric[platform]'"
        )
