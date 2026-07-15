#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TORCH_PROFILE="${TRIQTO_TORCH_PROFILE:-cuda}"
TORCH_CUDA_INDEX="${TRIQTO_TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"

cd "${REPO_ROOT}"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"TriQTO requires Python 3.11.x, found {sys.version.split()[0]}")
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -c constraints/cpu.txt

if [[ "${TORCH_PROFILE}" == "cuda" ]]; then
  python -m pip uninstall -y torch
  python -m pip install --index-url "${TORCH_CUDA_INDEX}" "torch==2.8.0"
elif [[ "${TORCH_PROFILE}" != "cpu" ]]; then
  echo "TRIQTO_TORCH_PROFILE must be 'cpu' or 'cuda'" >&2
  exit 2
fi

python -m pip install -e .
python scripts/verify_dependency_pins.py

python - <<'PY'
import json
import torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
}, sort_keys=True))
PY

echo "TriQTO Phase 15.6 pod environment installed."
echo "Next: python scripts/run_phase15_6_campaign.py prepare --workspace /workspace/triqto-data"
