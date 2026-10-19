"""Content-addressed finite-shot evidence. No inference, fitting, or truth access."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

SCHEMA = "triqto.finite_shot_counts.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def array_identity(value):
    a = np.ascontiguousarray(value)
    return {"dtype": str(a.dtype), "shape": list(a.shape), "sha256": sha(a.tobytes())}


def circuit_identity(circuit):
    """Bind operation order, wires, parameters, and phase without QPY UUIDs."""
    return {
        "qubits": circuit.num_qubits, "clbits": circuit.num_clbits,
        "global_phase": str(circuit.global_phase),
        "operations": [
            {"name": item.operation.name,
             "params": [str(p) for p in item.operation.params],
             "qubits": [circuit.find_bit(q).index for q in item.qubits],
             "clbits": [circuit.find_bit(c).index for c in item.clbits]}
            for item in circuit.data
        ],
    }


def probability_difference(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("invalid probability comparison")
    delta = np.abs(a - b)
    return {"max_abs": float(delta.max(initial=0)),
            "l1": float(delta.sum()), "total_variation": float(delta.sum() / 2),
            "different_elements": int(np.count_nonzero(delta)),
            "elements": int(a.size),
            "mass_error_a": float(abs(a.sum() - 1)),
            "mass_error_b": float(abs(b.sum() - 1))}


class CalibrationBundle:
    """Capture once; replay counts only after checking every call's provenance."""

    def __init__(self, source=None, expected_sha256=None):
        self.source = Path(source) if source else None
        self.roots = {}
        self.arrays = {}
        self.seen = {"phasea": set(), "posthoc": set()}
        self.active = None
        self.document = None
        if self.source:
            raw = (self.source / "manifest.json").read_bytes()
            if not expected_sha256 or sha(raw) != expected_sha256:
                raise ValueError("calibration manifest hash mismatch")
            doc = json.loads(raw)
            if doc["schema"] != SCHEMA:
                raise ValueError("unsupported calibration bundle schema")
            data = (self.source / "arrays.npz").read_bytes()
            if sha(data) != doc["arrays_sha256"]:
                raise ValueError("calibration array archive hash mismatch")
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                self.arrays = {k: archive[k] for k in archive.files}
            self.document = doc
            self.roots = doc["roots"]
            for root, rec in self.roots.items():
                p, c = self.arrays["p" + root], self.arrays["c" + root]
                if p.shape != c.shape or len(p) != len(rec["calls"]):
                    raise ValueError("bundle array/call shape mismatch")
                if p.dtype != np.dtype("<f8") or c.dtype != np.dtype("<i8"):
                    raise ValueError("noncanonical bundle dtypes")
                if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(c < 0):
                    raise ValueError("invalid probabilities or sample counts")
                for index, call in enumerate(rec["calls"]):
                    if int(c[index].sum()) != call["shots"]:
                        raise ValueError("bundle sample-count shot total mismatch")

    def begin(self, root, context, path):
        root = str(root)
        if self.active is not None or root in self.seen[path]:
            raise ValueError("duplicate or nested calibration root")
        if self.source or path != "phasea":
            if root not in self.roots or self.roots[root]["context"] != context:
                raise ValueError("calibration circuit/configuration provenance mismatch")
        else:
            self.roots[root] = {"context": context, "calls": []}
            self.arrays["p" + root], self.arrays["c" + root] = [], []
        self.active = (root, path)
        self.cursor = 0

    def sample(self, probabilities, shots, eig, pairs, seed, native_counts):
        if self.active is None:
            raise ValueError("sample outside active root")
        root, path = self.active
        call = {"ordinal": self.cursor, "seed": int(seed), "shots": int(shots),
                "probability_shape": list(np.shape(probabilities)),
                "measurement_eigenvalues": array_identity(eig),
                "ordered_pairs": array_identity(pairs),
                # finite_shot_candidate_jacobians loops candidate, mechanism,
                # sign, then the declared basis order; this ordering is also bound in root context.
                "candidate_index": self.cursor // 18,
                "mechanism_index": (self.cursor // 6) % 3,
                "sign": ["plus", "minus"][(self.cursor // 3) % 2],
                "basis": self.roots[root]["context"]["basis_order"][self.cursor % 3]}
        rec = self.roots[root]
        if not self.source and path == "phasea":
            rec["calls"].append(call)
            self.arrays["p" + root].append(np.array(probabilities, dtype="<f8", copy=True))
            self.arrays["c" + root].append(np.array(native_counts, dtype="<i8", copy=True))
            counts = native_counts
        else:
            if self.cursor >= len(rec["calls"]) or rec["calls"][self.cursor] != call:
                raise ValueError("calibration sampling/measurement provenance mismatch")
            counts = self.arrays["c" + root][self.cursor]
            if not self.source and (not np.array_equal(counts, native_counts) or
                    not np.array_equal(self.arrays["p" + root][self.cursor], probabilities)):
                raise ValueError("capture extraction-path evidence mismatch")
        self.cursor += 1
        return counts

    def end(self):
        root, path = self.active
        if self.cursor != len(self.roots[root]["calls"]):
            raise ValueError("calibration call coverage mismatch")
        self.seen[path].add(root)
        self.active = None

    def verify_coverage(self):
        if self.active is not None or any(v != set(self.roots) for v in self.seen.values()):
            raise ValueError("calibration root coverage mismatch")

    def save(self, parent, provenance):
        self.verify_coverage()
        if self.source:
            if self.document["provenance"] != provenance:
                raise ValueError("calibration scientific source provenance mismatch")
            return {"directory": str(self.source), "manifest_sha256": sha((self.source / "manifest.json").read_bytes()),
                    "mode": "substitute_counts", "schema": SCHEMA}
        stream = io.BytesIO()
        np.savez_compressed(stream, **{k: np.asarray(v) for k, v in self.arrays.items()})
        data = stream.getvalue()
        doc = {"schema": SCHEMA, "arrays_sha256": sha(data), "roots": self.roots,
               "provenance": provenance,
               "count_semantics": "integer multinomial outcomes; stats reconstructed locally; no sampler on replay"}
        raw = canonical(doc)
        digest = sha(raw)
        directory = Path(parent) / digest.removeprefix("sha256:")
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "arrays.npz").write_bytes(data)
        (directory / "manifest.json").write_bytes(raw)
        return {"directory": str(directory), "manifest_sha256": digest, "mode": "capture", "schema": SCHEMA}


def compare_bundles(left, left_hash, right, right_hash):
    a, b = CalibrationBundle(left, left_hash), CalibrationBundle(right, right_hash)
    if a.document["provenance"] != b.document["provenance"] or set(a.roots) != set(b.roots):
        raise ValueError("different scientific provenance or circuit cohort")
    roots = {}
    for root in sorted(a.roots):
        if a.roots[root] != b.roots[root]:
            raise ValueError("different root or sampling provenance")
        pa, pb = a.arrays["p" + root], b.arrays["p" + root]
        ca, cb = a.arrays["c" + root], b.arrays["c" + root]
        calls = [probability_difference(x, y) for x, y in zip(pa, pb)]
        roots[root] = {
            "max_probability_abs": max(c["max_abs"] for c in calls),
            "max_probability_l1": max(c["l1"] for c in calls),
            "max_probability_tv": max(c["total_variation"] for c in calls),
            "probability_different_calls": sum(c["different_elements"] > 0 for c in calls),
            "count_different_calls": int(np.any(ca != cb, axis=1).sum()),
            "max_count_abs": int(np.max(np.abs(ca - cb))),
            "calls": calls,
        }
    return {"schema": "triqto.calibration_numeric_comparison.v1", "per_root": roots,
            "root_count": len(roots),
            "max_probability_abs": max(v["max_probability_abs"] for v in roots.values()),
            "max_probability_tv": max(v["max_probability_tv"] for v in roots.values()),
            "count_different_roots": sum(v["count_different_calls"] > 0 for v in roots.values())}
