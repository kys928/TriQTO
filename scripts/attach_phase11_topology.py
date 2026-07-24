#!/usr/bin/env python3
"""Environment-variable runner for split-safe Phase 11 topology attachment."""
from __future__ import annotations

import json
import sys

from triqto.phase15_6.topology_attachment import (
    TopologyAttachmentConfig,
    attach_phase11_topology,
)


def main() -> int:
    try:
        config = TopologyAttachmentConfig.from_environment()
        print("=" * 88)
        print("TRIQTO PHASE 11 → PHASE 12 TOPOLOGY ATTACHMENT")
        print("=" * 88)
        print(f"Phase 11:            {config.phase11_root}")
        print(f"Phase 12:            {config.phase12_root}")
        print(f"Model-ready source:  {config.model_ready_root}")
        print(f"Output base:         {config.output_root}")
        print(f"Copy mode:           {config.copy_mode}")
        print(f"Strict:              {config.strict}")
        print(f"Attach hardware:     {config.attach_hardware_masked}")
        print(f"Joint diagnosis use: {config.enable_joint_diagnosis}")
        print("Topology loss:       0.0")
        print(f"Dry run:             {config.dry_run}")
        print()
        result = attach_phase11_topology(config)
        print()
        print("=" * 88)
        print("TOPOLOGY ATTACHMENT COMPLETE")
        print("=" * 88)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
