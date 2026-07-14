"""Phase 15 integration for operational actions and checkpoint-bound topology.

The existing evaluator remains authoritative for trained-model metrics. This
wrapper appends family-specific operational evidence and diagnostic topology
without pooling incomparable rewards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import shutil
import uuid

import yaml
from triqto.core.ids import canonical_json, make_deterministic_id
from .evaluator import Phase15EvaluationConfig, run_phase15_evaluation
from .operational_reports import summarize_checkpoint_latent_topology, summarize_operational_actions

INTEGRATED_PHASE15_SCHEMA = "triqto.phase15.operational_topology.v1"


@dataclass(frozen=True, slots=True)
class IntegratedPhase15Config:
    core_config: str = "configs/eval/phase15_smoke.yaml"
    require_operational_actions: bool = True
    require_checkpoint_latent_topology: bool = True
    label: str = "operational and checkpoint-latent engineering validation"

    def __post_init__(self) -> None:
        if not self.core_config.strip() or not self.label.strip():
            raise ValueError("integrated Phase 15 text fields must be nonblank")
        if not isinstance(self.require_operational_actions, bool) or not isinstance(self.require_checkpoint_latent_topology, bool):
            raise TypeError("integrated Phase 15 requirements must be bool")
        if "research" in self.label.lower():
            raise ValueError("integrated Phase 15 must not claim research evidence")


def load_integrated_phase15_config(path: str | Path) -> IntegratedPhase15Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("integrated Phase 15 config must contain mapping")
    allowed = set(IntegratedPhase15Config.__dataclass_fields__)  # type: ignore[attr-defined]
    if set(payload) - allowed:
        raise ValueError(f"unknown integrated Phase 15 fields: {sorted(set(payload) - allowed)}")
    return IntegratedPhase15Config(**dict(payload))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(namespace: str, payload: Mapping[str, Any]) -> str:
    return make_deterministic_id(namespace, {"payload": canonical_json(dict(payload))})


def run_integrated_phase15_evaluation(*, training_view_root: str | Path, training_root: str | Path, checkpoint: str | Path, output_root: str | Path, core_config: Phase15EvaluationConfig, integration_config: IntegratedPhase15Config, operational_action_root: str | Path | None, latent_topology_root: str | Path | None, phase7_root: str | Path | None = None) -> dict[str, Any]:
    if integration_config.require_operational_actions and operational_action_root is None:
        raise ValueError("integrated Phase 15 requires operational actions")
    if integration_config.require_checkpoint_latent_topology and latent_topology_root is None:
        raise ValueError("integrated Phase 15 requires checkpoint-bound topology")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"integrated Phase 15 output exists: {output}")
    core_temp = output.parent / f".{output.name}.core-{uuid.uuid4().hex}"
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        core = run_phase15_evaluation(training_view_root=training_view_root, training_root=training_root, checkpoint=checkpoint, output_root=core_temp, config=core_config, phase7_root=phase7_root)
        operational = summarize_operational_actions(operational_action_root) if operational_action_root is not None else None
        latent = summarize_checkpoint_latent_topology(latent_topology_root) if latent_topology_root is not None else None
        identity = {
            "schema": INTEGRATED_PHASE15_SCHEMA,
            "core_phase15_run_id": core["summary"]["phase15_run_id"],
            "checkpoint_id": core["summary"]["checkpoint_id"],
            "operational_action_dataset_id": operational["operational_action_dataset_id"] if operational else None,
            "latent_topology_id": latent["latent_topology_id"] if latent else None,
            "integration_config": asdict(integration_config),
        }
        integrated_id = make_deterministic_id("phase15_integrated", identity)
        summary = dict(core["summary"])
        summary.update({
            "schema_version": INTEGRATED_PHASE15_SCHEMA,
            "integrated_phase15_run_id": integrated_id,
            "core_phase15_run_id": core["summary"]["phase15_run_id"],
            "integration_label": integration_config.label,
            "operational_action_reporting": operational,
            "checkpoint_latent_topology_reporting": latent,
            "action_family_metrics_pooled": False,
            "latent_topology_diagnostic_only": True,
            "physical_hardware": False,
            "topology_loss_weight": 0.0,
            "claim_scope": "CPU smoke engineering validation with separate operational metrics and diagnostic checkpoint-bound topology; not research evidence",
        })
        summary.pop("summary_content_hash", None)
        summary["summary_content_hash"] = _hash("phase15_integrated_summary", summary)
        card = dict(core["card"])
        card.update({
            "integrated_phase15_run_id": integrated_id,
            "operational_action_dataset_id": operational["operational_action_dataset_id"] if operational else None,
            "latent_topology_id": latent["latent_topology_id"] if latent else None,
            "action_families_reported_separately": True,
            "topology_diagnostic_only": True,
        })
        card.pop("card_content_hash", None)
        limitations = list(card.get("limitations", [])) + [
            "basis probes acquire evidence and are not corrections",
            "layout/routing/depth metrics are not pooled with logical-correction reward",
            "latent topology is diagnostic and demonstrates no topology benefit",
            "topology loss remains exactly zero",
        ]
        card["limitations"] = sorted(set(limitations))
        card["card_content_hash"] = _hash("phase15_integrated_card", card)
        manifest = {
            "schema_version": INTEGRATED_PHASE15_SCHEMA,
            "integrated_phase15_run_id": integrated_id,
            "core_phase15_run_id": core["summary"]["phase15_run_id"],
            "summary_content_hash": summary["summary_content_hash"],
            "card_content_hash": card["card_content_hash"],
            "managed_files": ["phase15_integrated_summary.json", "phase15_integrated_card.json", "phase15_integrated_complete.json"],
        }
        manifest["manifest_content_hash"] = _hash("phase15_integrated_manifest", manifest)
        staging.mkdir(parents=True)
        _write_json(staging / "phase15_integrated_summary.json", summary)
        _write_json(staging / "phase15_integrated_card.json", card)
        _write_json(staging / "phase15_integrated_complete.json", manifest)
        staging.replace(output)
        return {"summary": summary, "card": card, "manifest": manifest}
    finally:
        if core_temp.exists():
            shutil.rmtree(core_temp)
        if staging.exists():
            shutil.rmtree(staging)


def load_integrated_phase15_result(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    manifest = json.loads((base / "phase15_integrated_complete.json").read_text(encoding="utf-8"))
    manifest_hash = manifest.pop("manifest_content_hash", None)
    if manifest_hash != _hash("phase15_integrated_manifest", manifest):
        raise ValueError("integrated Phase 15 manifest content hash mismatch")
    manifest["manifest_content_hash"] = manifest_hash
    actual = {path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file()}
    if actual != set(manifest.get("managed_files", [])):
        raise ValueError("integrated Phase 15 managed inventory mismatch")
    summary = json.loads((base / "phase15_integrated_summary.json").read_text(encoding="utf-8"))
    summary_hash = summary.pop("summary_content_hash", None)
    if summary_hash != _hash("phase15_integrated_summary", summary):
        raise ValueError("integrated Phase 15 summary content hash mismatch")
    summary["summary_content_hash"] = summary_hash
    card = json.loads((base / "phase15_integrated_card.json").read_text(encoding="utf-8"))
    card_hash = card.pop("card_content_hash", None)
    if card_hash != _hash("phase15_integrated_card", card):
        raise ValueError("integrated Phase 15 card content hash mismatch")
    card["card_content_hash"] = card_hash
    if summary.get("topology_loss_weight") != 0.0 or summary.get("action_family_metrics_pooled") is not False:
        raise ValueError("integrated Phase 15 claim boundary mismatch")
    return {"summary": summary, "card": card, "manifest": manifest}


__all__ = ["INTEGRATED_PHASE15_SCHEMA", "IntegratedPhase15Config", "load_integrated_phase15_config", "load_integrated_phase15_result", "run_integrated_phase15_evaluation"]
