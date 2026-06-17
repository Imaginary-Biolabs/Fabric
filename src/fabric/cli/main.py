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
from fabric.utils.errors import FabricError

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
        if mode == "remote":
            from fabric.platform.runner import run_workflow_remote

            run = run_workflow_remote(workflow, inputs=_parse_set_pairs(set_), root=root)
        else:
            remote_submit = None
            if mode in {"remote", "hybrid"}:
                from fabric.platform.runner import make_remote_submit

                remote_submit = make_remote_submit()
            run = Runner(mode=mode, root=root, remote_submit=remote_submit).run(
                workflow, inputs=_parse_set_pairs(set_)
            )
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
    """Show platform API connectivity."""
    from fabric.platform.client import PlatformClient

    try:
        client = PlatformClient()
        health = client.request("GET", "/health")
        info_panel(
            f"Connected to [bold]{client.base_url}[/]\n"
            f"API version: {health.get('api_version', 'unknown')}",
            title="platform",
        )
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


upload_app = typer.Typer(name="upload", help="Upload dataset releases and model checkpoints.")
job_app = typer.Typer(name="job", help="Submit and monitor platform jobs.")
publish_app = typer.Typer(name="publish", help="Publish models and versions to the registry.")


@upload_app.command("release")
def platform_upload_release_cmd(
    asset: Annotated[str, typer.Option("--asset", "-a", help="Dataset asset id, e.g. D_000001")],
    version: Annotated[str, typer.Option("--version", "-v", help="Asset version")] = "1",
    path: Annotated[Path, typer.Option("--path", "-p", help="Released Zarr directory")] = ...,
) -> None:
    """Upload a Grumpy Zarr release directory."""
    from fabric.platform.upload import upload_release

    try:
        manifest = upload_release(asset_id=asset, version=version, path=path)
        success_panel(
            "upload complete",
            [
                ("kind", "dataset_release"),
                ("asset", asset),
                ("version", version),
                ("manifest_id", str(manifest.get("id", "—"))),
                ("objects", str(manifest.get("object_count", "—"))),
            ],
        )
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


@upload_app.command("checkpoint")
def platform_upload_checkpoint_cmd(
    asset: Annotated[str, typer.Option("--asset", "-a", help="Model asset id, e.g. M_000003")],
    version: Annotated[str, typer.Option("--version", "-v", help="Asset version")] = "1",
    path: Annotated[Path, typer.Option("--path", "-p", help="Checkpoint .pt or directory")] = ...,
) -> None:
    """Upload a model checkpoint.pt manifest."""
    from fabric.platform.upload import upload_checkpoint

    try:
        manifest = upload_checkpoint(asset_id=asset, version=version, path=path)
        success_panel(
            "upload complete",
            [
                ("kind", "model_checkpoint"),
                ("asset", asset),
                ("version", version),
                ("manifest_id", str(manifest.get("id", "—"))),
            ],
        )
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


@job_app.command("submit")
def platform_job_submit_cmd(
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model asset id for workflow_run")] = None,
    mode: Annotated[str, typer.Option("--mode", help="workflow_run mode: run or develop")] = "run",
    asset_version: Annotated[str, typer.Option("--asset-version", help="Model version")] = "1",
    develop_profile: Annotated[
        str | None, typer.Option("--develop-profile", help="Develop profile id")
    ] = None,
    inputs: Annotated[
        str | None, typer.Option("--inputs", help="JSON object of run/develop inputs")
    ] = None,
    inputs_file: Annotated[
        Path | None, typer.Option("--inputs-file", help="Path to JSON inputs file")
    ] = None,
    parameters: Annotated[
        str | None, typer.Option("--parameters", help="JSON object of execution parameters")
    ] = None,
    wait: Annotated[bool, typer.Option("--wait", help="Block until job completes.")] = False,
    benchmark: Annotated[
        str | None, typer.Option("--benchmark", "-b", help="Benchmark id (legacy eval shim)")
    ] = None,
    benchmark_version: Annotated[str, typer.Option("--benchmark-version")] = "1",
    model_version: Annotated[str, typer.Option("--model-version")] = "1",
    batch_size: Annotated[int, typer.Option("--batch-size")] = 8,
) -> None:
    """Submit a workflow_run job or legacy benchmark_eval shim."""
    from fabric.platform.jobs import (
        submit_benchmark_eval,
        submit_workflow_run,
        wait_for_job,
    )

    try:
        if benchmark:
            if model is None:
                raise typer.BadParameter("--model is required with --benchmark")
            job = submit_benchmark_eval(
                benchmark_id=benchmark,
                benchmark_version=benchmark_version,
                model_id=model,
                model_version=model_version,
                overrides={"batch_size": batch_size, "split": "test"},
            )
        elif model:
            parsed_inputs: dict[str, Any] = {}
            if inputs_file is not None:
                parsed_inputs = json.loads(inputs_file.read_text())
            elif inputs:
                parsed_inputs = json.loads(inputs)
            parsed_parameters: dict[str, Any] = json.loads(parameters) if parameters else {}
            job = submit_workflow_run(
                asset_id=model,
                version=asset_version,
                mode=mode,
                inputs=parsed_inputs,
                parameters=parsed_parameters,
                develop_profile=develop_profile,
            )
        else:
            raise typer.BadParameter("Provide --model for workflow_run or --benchmark for eval shim")

        if wait:
            job = wait_for_job(job["id"])

        rows = [
            ("job_id", job["id"]),
            ("status", job["status"]),
            ("type", job.get("type", "workflow_run")),
        ]
        if job.get("result"):
            rows.append(("result", json.dumps(job["result"], sort_keys=True)))
        success_panel("job submitted", rows)
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


@job_app.command("materialize-designs")
def platform_job_materialize_designs_cmd(
    job_id: Annotated[str, typer.Argument(help="Completed workflow job UUID")],
    candidates: Annotated[
        str | None,
        typer.Option("--candidates", help="Comma-separated candidate ids"),
    ] = None,
) -> None:
    """Materialize design candidates from a completed workflow job."""
    from fabric.platform.publish import materialize_designs

    candidate_ids = [item.strip() for item in candidates.split(",") if item.strip()] if candidates else None
    try:
        result = materialize_designs(job_id=job_id, candidate_ids=candidate_ids)
        success_panel(
            "designs materialized",
            [
                ("job_id", job_id),
                ("count", str(len(result.get("promoted_designs", [])))),
                ("result", json.dumps(result, sort_keys=True)),
            ],
        )
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


@job_app.command("status")
def platform_job_status_cmd(
    job_id: Annotated[str, typer.Argument(help="Job UUID")],
    wait: Annotated[bool, typer.Option("--wait", help="Block until job completes.")] = False,
) -> None:
    """Get or wait for job status."""
    from fabric.platform.jobs import get_job, wait_for_job

    try:
        job = wait_for_job(job_id) if wait else get_job(job_id)
        rows = [("job_id", job_id), ("status", job["status"])]
        if job.get("result"):
            rows.append(("result", json.dumps(job["result"], sort_keys=True)))
        success_panel("job status", rows)
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


@publish_app.command("design")
def platform_publish_design_cmd(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ],
    title: Annotated[str | None, typer.Option("--title", help="Display title")] = None,
    visibility: Annotated[str, typer.Option("--visibility", help="private or public")] = "private",
) -> None:
    """Publish a new design asset from a local YAML config."""
    from fabric.platform.publish import publish_design

    try:
        created = publish_design(config_path=config, title=title, visibility=visibility)
        success_panel(
            "design published",
            [
                ("asset_id", str(created.get("id", "—"))),
                ("version", str(created.get("version", "1"))),
                ("visibility", visibility),
            ],
        )
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


@publish_app.command("workflow")
def platform_publish_workflow_cmd(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ],
    title: Annotated[str | None, typer.Option("--title", help="Display title")] = None,
    visibility: Annotated[str, typer.Option("--visibility", help="private or public")] = "private",
    run_price_micros: Annotated[
        int | None, typer.Option("--run-price-micros", help="Per-run price in micros")
    ] = None,
) -> None:
    """Publish a new workflow asset from a local YAML config."""
    from fabric.platform.publish import publish_workflow

    try:
        created = publish_workflow(
            config_path=config,
            title=title,
            visibility=visibility,
            run_price_micros=run_price_micros,
        )
        success_panel(
            "workflow published",
            [
                ("asset_id", str(created.get("id", "—"))),
                ("version", str(created.get("version", "1"))),
                ("visibility", visibility),
            ],
        )
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


@publish_app.command("model")
def platform_publish_model_cmd(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ],
    title: Annotated[str | None, typer.Option("--title", help="Display title")] = None,
    visibility: Annotated[str, typer.Option("--visibility", help="private or public")] = "private",
) -> None:
    """Publish a new model asset from a local YAML config."""
    from fabric.platform.publish import publish_model

    try:
        created = publish_model(config_path=config, title=title, visibility=visibility)
        success_panel(
            "model published",
            [
                ("asset_id", str(created.get("id", "—"))),
                ("version", str(created.get("version", "1"))),
                ("visibility", visibility),
            ],
        )
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


@publish_app.command("model-version")
def platform_publish_model_version_cmd(
    base: Annotated[str, typer.Option("--base", "-b", help="Base model asset id")],
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", "-p", exists=True, help="Checkpoint .pt or directory"),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Version config YAML (defaults to latest)"),
    ] = None,
    version: Annotated[str | None, typer.Option("--version", "-v")] = None,
    visibility: Annotated[str, typer.Option("--visibility", help="private or public")] = "private",
) -> None:
    """Publish a new model version with checkpoint upload."""
    from fabric.platform.publish import publish_model_version

    try:
        result = publish_model_version(
            asset_id=base,
            config_path=config,
            checkpoint_path=checkpoint,
            version=version,
            visibility=visibility,
        )
        success_panel(
            "model version published",
            [
                ("asset_id", base),
                ("version", str(result["version"])),
                ("manifest_id", str(result["manifest"].get("id", "—"))),
                ("visibility", visibility),
            ],
        )
    except FabricError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc


platform_app.add_typer(upload_app)
platform_app.add_typer(job_app)
platform_app.add_typer(publish_app)


app.add_typer(workflow_app)
app.add_typer(platform_app)


def main_entry() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    main_entry()
