#!/usr/bin/env python3
"""Verified launcher for the frozen confirmatory holdout generator.

The full implementation is stored as a compressed UTF-8 payload so the exact
source can be hash-bound by the protocol freeze. This launcher verifies the
uncompressed source digest before executing it.
"""
from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

PAYLOAD = (
    Path(__file__).resolve().parent
    / "confirmatory_payloads"
    / "generate_phase_amplitude_confirmatory_holdout.py.zlib.b64"
)
EXPECTED_SOURCE_SHA256 = (
    "08edd1a77eb0e13cef5e20253c66ba5a58ffb319190d8ca8f6f0f21d4a90949b"
)


def main() -> None:
    encoded = PAYLOAD.read_text(encoding="ascii").strip()
    source = zlib.decompress(base64.b64decode(encoded, validate=True))
    actual = hashlib.sha256(source).hexdigest()
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Confirmatory holdout generator payload hash mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, found {actual}"
        )
    filename = str(Path(__file__).resolve())
    namespace = {
        "__name__": "__main__",
        "__file__": filename,
        "__package__": None,
    }
    exec(compile(source, filename, "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
