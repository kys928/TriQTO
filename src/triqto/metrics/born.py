"""Born-probability metric placeholders for TriQTO."""
from __future__ import annotations

def total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Future total variation distance over Born distributions."""
    raise NotImplementedError("Phase 6 will implement Born metrics.")
def safe_kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Future numerically safe KL divergence."""
    raise NotImplementedError("Phase 6 will implement Born metrics.")
def hellinger_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Future Hellinger distance."""
    raise NotImplementedError("Phase 6 will implement Born metrics.")
def jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Future Jensen-Shannon divergence."""
    raise NotImplementedError("Phase 6 will implement Born metrics.")
