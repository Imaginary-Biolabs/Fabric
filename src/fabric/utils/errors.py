"""Fabric exception types."""


class FabricError(Exception):
    """Base class for all Fabric errors."""


class ConfigurationError(FabricError):
    """Invalid or missing user or runtime configuration.

    Raised when settings, YAML paths, or general configuration is invalid.
    """


class ConfigError(ConfigurationError):
    """Invalid dataset or factory configuration.

    Raised when a dataset YAML declares conflicting or incomplete options.
    """


class ParseError(FabricError):
    """Structure file parsing failed.

    Raised when a PDB or mmCIF file cannot be read or contains no atoms.
    """


class ExternalError(FabricError):
    """External data source adapter failed.

    Raised when glob patterns match no files or an adapter is misconfigured.
    """


class DatasetError(FabricError):
    """Dataset release or cache error.

    Raised when a dataset is accessed before release or a parent is unavailable.
    """


class SchemaError(FabricError):
    """Biological schema level missing or invalid.

    Raised when Grumpy data does not conform to :data:`~fabric.utils.constants.SCHEMA`.
    """


class TransformError(FabricError):
    """Transform pipeline error.

    Raised when transform configuration or execution fails.
    """


class BenchmarkError(FabricError):
    """Benchmark configuration or loader error."""


class SamplerError(FabricError):
    """Sampler configuration or indexing error."""


class TaskError(FabricError):
    """Task input/target extraction error."""


class MetricError(FabricError):
    """Metric state or computation error."""


class CollateError(FabricError):
    """Batch collation error."""


class BackendError(FabricError):
    """Deep learning backend configuration or execution error."""


class TrainerError(FabricError):
    """Trainer configuration or loop error."""


class ModelError(FabricError):
    """Model, layer, scaffold, or objective configuration error."""


class WorkflowError(FabricError):
    """Workflow specification, planning, or execution error."""


class RunnerError(FabricError):
    """Runner configuration or execution error."""


class BackendExtraRequired(FabricError):
    """Optional backend dependencies are not installed."""

    def __init__(self, extra: str) -> None:
        super().__init__(
            f"Backend features require the {extra} extra: pip install 'imaginary-fabric[{extra}]'"
        )


class PlatformExtraRequired(FabricError):
    """Optional platform dependencies are not installed.

    Install with ``pip install 'imaginary-fabric[platform]'`` to use
    :mod:`fabric.platform` HTTP clients.
    """

    def __init__(self) -> None:
        super().__init__(
            "Platform features require the platform extra: pip install 'imaginary-fabric[platform]'"
        )
