"""Competition team registration and artifact submission."""

from __future__ import annotations

from typing import Any

from fabric.platform.client import PlatformClient
from fabric.utils.errors import FabricError


def create_team(*, competition_id: str, name: str) -> dict[str, Any]:
    client = PlatformClient()
    return client.request(
        "POST",
        f"/competitions/{competition_id}/teams",
        json={"name": name},
    )


def list_my_teams(*, competition_id: str) -> dict[str, Any]:
    client = PlatformClient()
    return client.request("GET", f"/competitions/{competition_id}/teams/mine")


def submit_artifact(
    *,
    competition_id: str,
    team_id: str,
    artifact_id: str,
    asset_version: str = "1",
) -> dict[str, Any]:
    client = PlatformClient()
    return client.request(
        "POST",
        f"/competitions/{competition_id}/submissions",
        json={
            "team_id": team_id,
            "artifact_id": artifact_id,
            "asset_version": asset_version,
        },
    )


def list_my_submissions(*, competition_id: str) -> dict[str, Any]:
    client = PlatformClient()
    return client.request("GET", f"/competitions/{competition_id}/submissions/mine")


def advance_phase(
    *,
    competition_id: str,
    phase: str | None = None,
) -> dict[str, Any]:
    client = PlatformClient()
    payload = {"phase": phase} if phase else {}
    return client.request(
        "POST",
        f"/competitions/{competition_id}/advance-phase",
        json=payload,
    )


def get_leaderboard(
    *,
    competition_id: str,
    kind: str = "open",
) -> dict[str, Any]:
    client = PlatformClient()
    return client.request(
        "GET",
        f"/competitions/{competition_id}/leaderboard",
        params={"kind": kind},
    )


def submit_to_competition(
    *,
    competition_id: str,
    artifact_id: str,
    team_name: str | None = None,
    team_id: str | None = None,
    asset_version: str = "1",
) -> dict[str, Any]:
    """Register or reuse a team, then submit an artifact."""
    if team_id is None:
        if not team_name:
            raise FabricError("Provide --team-id or --team-name")
        created = create_team(competition_id=competition_id, name=team_name)
        team_id = str(created["team"]["id"])
    else:
        teams = list_my_teams(competition_id=competition_id)
        known = {str(item["id"]) for item in teams.get("items", [])}
        if team_id not in known:
            raise FabricError(f"Team {team_id} is not one of your teams for {competition_id}")

    return submit_artifact(
        competition_id=competition_id,
        team_id=team_id,
        artifact_id=artifact_id,
        asset_version=asset_version,
    )
