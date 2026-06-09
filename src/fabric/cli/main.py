"""Imaginary Fabric CLI entrypoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from fabric import Settings, __version__
from fabric.cli.console import (
    branded_result_table,
    console,
    info_panel,
    print_banner,
    print_error,
    success_panel,
)
from fabric.core.factory import Factory
from fabric.core.runner import Runner
from fabric.utils.errors import FabricError, PlatformExtraRequired

app = typer.Typer(
    name="imaginary",
    help="[bold]Imaginary Fabric[/] — molecular ML orchestration on Grumpy.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
)
workflow_app = typer.Typer(
    name="workflow",
    help="Run YAML workflow graphs locally or on hybrid compute.",
    no_args_is_help=True,
)
platform_app = typer.Typer(
    name="platform",
    help="Imaginary cloud platform commands (Phase 8).",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        print_banner(compact=True)
        typer.echo(f"imaginary {__version__}")
        raise typer.Exit()


def _parse_set_pairs(pairs: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` CLI overrides into a mapping."""
    parsed: dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise typer.BadParameter(f"Expected key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"Empty key in {item!r}")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed


def _with_settings(home: Path | None, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` inside an optional Settings context and map Fabric errors to exit codes."""
    try:
        if home is None:
            return fn()
        with Settings(home=home):
            return fn()
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the package version and exit.",
        ),
    ] = None,
) -> None:
    """Imaginary Fabric — AI infrastructure for programmable biology."""


@app.command("release")
def release_cmd(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Dataset YAML config.",
            show_default=False,
        ),
    ],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override Fabric data home (default: ~/.imaginary)."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if cache is valid.")] = False,
) -> None:
    """Release a dataset to Grumpy Zarr with provenance sidecars."""
    example = "tests/fixtures/datasets/D_local.yaml"

    def _run() -> None:
        dataset = Factory.dataset(config)
        path = dataset.release(force=force)
        meta = dataset.release_metadata()
        success_panel(
            "release complete",
            [
                ("dataset", f"{dataset.id} v{dataset.version}"),
                ("path", str(path)),
                ("transform_hash", str(meta.get("transform_hash", "—"))),
                ("molecules", str(meta.get("molecules", "—"))),
            ],
        )

    try:
        _with_settings(home, _run)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"{exc}\n\nExample: imaginary release --config {example}")
        raise typer.Exit(code=1) from exc


@app.command("train")
def train_cmd(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Trainer YAML config.",
            show_default=False,
        ),
    ],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override Fabric data home (default: ~/.imaginary)."),
    ] = None,
    epochs: Annotated[
        int | None,
        typer.Option("--epochs", help="Override training epochs from YAML."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="Override loader batch size from YAML."),
    ] = None,
    root: Annotated[
        str | None,
        typer.Option("--root", help="Override results directory from YAML."),
    ] = None,
    progress: Annotated[
        bool,
        typer.Option("--progress/--no-progress", help="Show loader bars."),
    ] = True,
) -> None:
    """Train a model on a benchmark using a YAML trainer config."""
    example = "tests/fixtures/train/train_mini.yaml"

    def _run() -> None:
        overrides: dict[str, Any] = {"progress": progress}
        if epochs is not None:
            overrides["epochs"] = epochs
        if batch_size is not None:
            overrides["batch_size"] = batch_size
        if root is not None:
            overrides["root"] = root
        trainer = Factory.trainer(str(config), **overrides)
        trainer.fit()
        checkpoint = trainer.root / f"{trainer.checkpoint_name}.pt"
        if not checkpoint.is_file():
            checkpoint = trainer.root / trainer.checkpoint_name
        success_panel(
            "training complete",
            [
                ("epochs", str(trainer.epochs)),
                ("steps", str(trainer.step)),
                ("root", str(trainer.root)),
                ("checkpoint", str(checkpoint)),
            ],
        )

    try:
        _with_settings(home, _run)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"{exc}\n\nExample: imaginary train --config {example} --epochs 2")
        raise typer.Exit(code=1) from exc


@app.command("eval")
def eval_cmd(
    benchmark: Annotated[
        str,
        typer.Option("--benchmark", "-b", help="Benchmark id or YAML path."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model id or YAML path."),
    ],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override Fabric data home (default: ~/.imaginary)."),
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Checkpoint stem or .pt path from training."),
    ] = None,
    train_config: Annotated[
        Path | None,
        typer.Option(
            "--train-config",
            help="Optional trainer YAML for collater and backend defaults.",
        ),
    ] = None,
    split: Annotated[str, typer.Option("--split", help="Benchmark split to evaluate.")] = "test",
    batch_size: Annotated[int, typer.Option("--batch-size", help="Loader batch size.")] = 8,
    progress: Annotated[
        bool,
        typer.Option("--progress/--no-progress", help="Show loader bars."),
    ] = False,
) -> None:
    """Evaluate a model on a benchmark and print a Rich metrics table."""
    example = (
        "imaginary eval --benchmark B_mini --model M_mini "
        "--checkpoint results/mini/checkpoint"
    )

    def _run() -> None:
        collater = None
        backend = None
        if train_config is not None:
            from omegaconf import OmegaConf

            from fabric.utils.config import load_config

            cfg = load_config(train_config)
            container = OmegaConf.to_container(cfg, resolve=True) or {}
            collater = container.get("collater")
            backend = container.get("backend")
        result = Factory.eval(
            benchmark,
            model,
            checkpoint=str(checkpoint) if checkpoint is not None else None,
            collater=collater,
            backend=backend,
            split=split,
            batch_size=batch_size,
            progress=progress,
        )
        console.print(branded_result_table(result.name, result.metrics))

    try:
        _with_settings(home, _run)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"{exc}\n\nExample: {example}")
        raise typer.Exit(code=1) from exc


@workflow_app.command("run")
def workflow_run_cmd(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Workflow YAML config.",
            show_default=False,
        ),
    ],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override Fabric data home (default: ~/.imaginary)."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Runner mode: local, hybrid, or remote."),
    ] = "local",
    root: Annotated[
        Path,
        typer.Option("--root", help="Directory for workflow run artifacts."),
    ] = Path("results/workflows"),
    set_: Annotated[
        list[str],
        typer.Option(
            "--set",
            "-s",
            help="Workflow input override as key=value (JSON values supported).",
        ),
    ] = [],
) -> None:
    """Execute a YAML workflow graph."""
    example = "tests/fixtures/workflows/W_mini.yaml"

    def _run() -> None:
        workflow = Factory.workflow(str(config))
        run = Runner(mode=mode, root=root).run(workflow, inputs=_parse_set_pairs(set_))
        rows = [
            ("status", run.status),
            ("run_id", run.run_id),
            ("root", str(Path(run.root) / run.run_id)),
        ]
        for key, value in run.outputs.items():
            if isinstance(value, dict):
                rows.append((key, json.dumps(value, sort_keys=True)))
            else:
                rows.append((key, str(value)))
        success_panel("workflow complete", rows)

    try:
        _with_settings(home, _run)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"{exc}\n\nExample: imaginary workflow run --config {example}")
        raise typer.Exit(code=1) from exc


@platform_app.command("status")
def platform_status_cmd() -> None:
    """Show platform integration status."""
    info_panel(
        "Imaginary Platform commands (upload, jobs, remote runner) ship in Phase 8.\n"
        "Install with: [bold]pip install 'imaginary-fabric[platform]'[/]",
        title="platform",
    )


@platform_app.command("upload")
def platform_upload_cmd() -> None:
    """Upload releases or checkpoints to Imaginary (Phase 8 stub)."""
    try:
        raise PlatformExtraRequired()
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


app.add_typer(workflow_app)
app.add_typer(platform_app)


def main_entry() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    main_entry()
