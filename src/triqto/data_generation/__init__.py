"""Phase 7 deterministic raw dataset generation."""
from .artifacts import verify_dataset_references, write_dataset
from .pipeline import generate_dataset
from .records import DatasetGenerationResult, DatasetWriteResult, GeneratedDatasetSample
from .seeding import derive_child_seed
from .specs import (
    CircuitGenerationSpec,
    DatasetGenerationConfig,
    DistortionSpec,
    config_from_dict,
    config_id,
    config_to_dict,
    load_generation_config,
    predicted_sample_count,
    save_generation_config,
    scientific_generation_id,
)

__all__ = [
    "CircuitGenerationSpec",
    "DatasetGenerationConfig",
    "DatasetGenerationResult",
    "DatasetWriteResult",
    "DistortionSpec",
    "GeneratedDatasetSample",
    "config_from_dict",
    "config_id",
    "config_to_dict",
    "derive_child_seed",
    "generate_dataset",
    "load_generation_config",
    "predicted_sample_count",
    "save_generation_config",
    "scientific_generation_id",
    "verify_dataset_references",
    "write_dataset",
]
