from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import zlib

import pytest


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (
        ROOT
        / "scripts/v0_2/confirmatory_payloads/"
        "generate_phase_amplitude_confirmatory_holdout.py.zlib.b64",
        "08edd1a77eb0e13cef5e20253c66ba5a58ffb319190d8ca8f6f0f21d4a90949b",
    ),
    (
        ROOT
        / "scripts/v0_2/confirmatory_payloads/"
        "evaluate_phase_amplitude_confirmatory_once.py.zlib.b64",
        "a3e6afc6f14314a6bc6371694a755547fa9e9b65b3bc0b4e504774202237206f",
    ),
)


@pytest.mark.parametrize(("payload", "expected_sha256"), CASES)
def test_confirmatory_payload_is_hash_bound_and_compilable(
    payload: Path,
    expected_sha256: str,
) -> None:
    encoded = payload.read_text(encoding="ascii").strip()
    source = zlib.decompress(base64.b64decode(encoded, validate=True))
    assert hashlib.sha256(source).hexdigest() == expected_sha256
    compile(source, str(payload.with_suffix("")), "exec")
