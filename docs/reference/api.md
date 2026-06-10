# API Reference

Reference documentation is generated from Python docstrings. For tutorials, start at [Home](../index.md).

## Factory

::: fabric.core.factory.Factory
    options:
      show_root_heading: true
      heading_level: 2
      members_order: source
      filters:
        - "!^__"

## Settings

::: fabric.utils.settings.Settings
    options:
      show_root_heading: true
      heading_level: 2
      filters:
        - "!^__"

## Dataset

::: fabric.core.dataset.Dataset
    options:
      show_root_heading: true
      heading_level: 2
      filters:
        - "!^__"

## Benchmark

::: fabric.core.benchmark.Benchmark
    options:
      show_root_heading: true
      heading_level: 2
      filters:
        - "!^__"

## Trainer

::: fabric.core.trainer.Trainer
    options:
      show_root_heading: true
      heading_level: 2
      filters:
        - "!^__"

## Workflow and Runner

::: fabric.core.workflow.Workflow
    options:
      show_root_heading: true
      heading_level: 2

::: fabric.core.runner.Runner
    options:
      show_root_heading: true
      heading_level: 2

## Platform client

::: fabric.platform.client.PlatformClient
    options:
      show_root_heading: true
      heading_level: 2

::: fabric.platform.registry
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - fetch_asset_version
        - fetch_asset
        - list_assets
        - cached_config_path

::: fabric.platform.jobs
    options:
      show_root_heading: true
      heading_level: 2

::: fabric.platform.upload
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - upload_release
        - upload_checkpoint

---

**Next:** [Architecture](../developer/architecture.md)
