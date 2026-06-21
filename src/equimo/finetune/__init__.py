"""Equinox-native fine-tuning scaffolding for Equimo."""

from ._typing import FilterSpec, LeafPredicate, Path, PyTree
from .config import (
    FineTuneBundle,
    FineTunePlan,
    GroupSpec,
    LLRDConfig,
    ParamInfo,
    TargetSpec,
    TrainableReport,
    TrainableSpec,
)
from .paths import (
    extract_param_paths,
    iter_param_leaves,
    iter_param_paths,
    make_param_info_tree,
    make_path_tree,
    path_to_str,
    str_to_path,
)
from .selectors import (
    is_layer_norm,
    is_linear,
    resolve_target,
    resolve_target_paths,
)
from .tags import (
    CANONICAL_TAGS,
    canonical_tags_for_path,
    infer_depth,
    iter_param_infos,
    make_tag_tree,
)

__all__ = (
    "CANONICAL_TAGS",
    "FilterSpec",
    "FineTuneBundle",
    "FineTunePlan",
    "GroupSpec",
    "LLRDConfig",
    "LeafPredicate",
    "ParamInfo",
    "Path",
    "PyTree",
    "TargetSpec",
    "TrainableReport",
    "TrainableSpec",
    "canonical_tags_for_path",
    "extract_param_paths",
    "infer_depth",
    "is_layer_norm",
    "is_linear",
    "iter_param_infos",
    "iter_param_leaves",
    "iter_param_paths",
    "make_param_info_tree",
    "make_path_tree",
    "make_tag_tree",
    "path_to_str",
    "resolve_target",
    "resolve_target_paths",
    "str_to_path",
)
