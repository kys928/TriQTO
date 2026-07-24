"""Phase 15.6 research-campaign preparation, execution, and aggregation."""
from .campaign import (
    aggregate_campaign,
    prepare_campaign,
    run_data_stage,
    run_evaluation_stage,
    run_training_stage,
)
from .config import (
    DataBuildConfig,
    PHASE156_CONFIG_SCHEMA,
    Phase156CampaignConfig,
    PodRequirements,
    load_phase156_config,
    phase156_config_from_dict,
    phase156_config_to_dict,
    save_phase156_config,
)
from .environment import inspect_phase156_environment
from .planner import PHASE156_PLAN_SCHEMA, build_campaign_plan, resolve_config_path
from .topology_attachment import (
    ATTACHMENT_VERSION,
    SCHEMA_VERSION as TOPOLOGY_ATTACHMENT_SCHEMA_VERSION,
    TopologyAttachmentConfig,
    attach_phase11_topology,
)

__all__ = [
    "ATTACHMENT_VERSION",
    "DataBuildConfig",
    "PHASE156_CONFIG_SCHEMA",
    "PHASE156_PLAN_SCHEMA",
    "Phase156CampaignConfig",
    "PodRequirements",
    "TOPOLOGY_ATTACHMENT_SCHEMA_VERSION",
    "TopologyAttachmentConfig",
    "aggregate_campaign",
    "attach_phase11_topology",
    "build_campaign_plan",
    "inspect_phase156_environment",
    "load_phase156_config",
    "phase156_config_from_dict",
    "phase156_config_to_dict",
    "prepare_campaign",
    "resolve_config_path",
    "run_data_stage",
    "run_evaluation_stage",
    "run_training_stage",
    "save_phase156_config",
]
