"""Fine-tuning delta serialization."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from importlib import metadata as package_metadata
from pathlib import Path
import tarfile
import tempfile
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lz4.frame
import numpy as np

from ._typing import PyTree
from .config import (
    FineTuneBundle,
    FineTuneBundleError,
    ModelLineage,
    ProjectionSegment,
    TargetSpec,
)
from .paths import key_path_to_path, path_to_str, str_to_path
from .peft.adapters import extract_adapter_delta, load_adapter_delta, strip_adapters
from .peft.base import get_path
from .peft.dora import DoRALinear, DoRAMergedLinear
from .peft.ia3 import IA3Linear
from .peft.lora import (
    extract_lora_delta,
    iter_lora_modules,
    load_lora_delta,
    strip_lora,
)
from .peft.lora import architecture_hash, merge_lora
from .peft.prefix import (
    PrefixConfig,
    PrefixProjection,
    PrefixTunedModel,
    strip_prefixes,
)
from .peft.prompts import (
    PTuningV2Config,
    PromptConfig,
    PromptedModel,
    SoftPromptConfig,
    VPTDeepConfig,
    VPTShallowConfig,
)
from .peft.scale_shift import ScaleShift, ScaleShiftWrapper
from .peft.vera import VeRALinear, strip_vera

_FORMAT = "equimo.finetune.bundle"
_FORMAT_VERSION = 1
_ARRAY_MARKER = "__equimo_finetune_array__"
_TUPLE_MARKER = "__equimo_finetune_tuple__"


def save_delta(
    *args,
    method: str = "lora",
    metadata: dict[str, Any] | None = None,
    model_state: Any | None = None,
    recalibration_required: bool | None = None,
    model: PyTree | None = None,
    path: str | Path | None = None,
    base_model: PyTree | None = None,
    spec: Any | None = None,
) -> FineTuneBundle:
    """Save a method delta bundle and return the saved bundle.

    ``save_delta(model, path, ...)``, ``save_delta(path, model, base_model, spec)``,
    and the spec-style ``save_delta(path, model=..., base_model=..., spec=...)``
    call orders are accepted. ``base_model`` and ``spec`` are metadata inputs;
    optimizers remain external. ``model_state`` must be a bundle-serializable
    snapshot when supplied; otherwise use ``recalibration_required=True`` for
    exports whose state must be recalibrated before evaluation.
    """

    model, path, resolved_base_model, resolved_spec = _resolve_save_delta_args(
        args,
        model=model,
        path=path,
        base_model=base_model,
        spec=spec,
    )

    if method == "lora":
        bundle = extract_lora_delta(model)
    elif method == "dora":
        bundle = _extract_dora_delta(model)
    elif method == "adapter":
        bundle = extract_adapter_delta(model)
    elif method == "prompt":
        bundle = _extract_prompt_delta(model)
    elif method == "prefix":
        bundle = _extract_prefix_delta(model)
    elif method == "scale_shift":
        bundle = _extract_scale_shift_delta(model)
    elif method == "ia3":
        bundle = _extract_ia3_delta(model)
    elif method == "vera":
        bundle = _extract_vera_delta(model)
    else:
        raise ValueError(
            "Unsupported delta method "
            f"{method!r}; currently 'lora', 'dora', 'adapter', 'prompt', "
            "'prefix', 'scale_shift', 'ia3', or 'vera'."
        )

    bundle = _enrich_bundle(
        bundle,
        model,
        base_model=resolved_base_model,
        spec=resolved_spec,
        user_metadata=metadata,
        model_state=model_state,
        recalibration_required=recalibration_required,
    )
    save_finetune_bundle(path, bundle)
    return bundle


def load_delta(
    *args,
) -> PyTree:
    """Load a delta bundle into a compatible base model.

    Both ``load_delta(base_model, path_or_bundle)`` and the spec-style
    ``load_delta(path_or_bundle, base_model)`` call order are accepted.
    """

    base_model, path_or_bundle = _resolve_load_delta_args(args)
    bundle = (
        path_or_bundle
        if isinstance(path_or_bundle, FineTuneBundle)
        else _read_bundle(path_or_bundle)
    )
    _check_schema(bundle)
    _check_base_checkpoint(base_model, bundle)
    _check_bundle_lineage_consistency(bundle)
    if bundle.method == "lora":
        return load_lora_delta(base_model, bundle)
    if bundle.method == "adapter":
        return load_adapter_delta(base_model, bundle)
    if bundle.method == "dora":
        return _load_dora_delta(base_model, bundle)
    if bundle.method == "prompt":
        return _load_prompt_delta(base_model, bundle)
    if bundle.method == "prefix":
        return _load_prefix_delta(base_model, bundle)
    if bundle.method == "scale_shift":
        return _load_scale_shift_delta(base_model, bundle)
    if bundle.method == "ia3":
        return _load_ia3_delta(base_model, bundle)
    if bundle.method == "vera":
        return _load_vera_delta(base_model, bundle)
    raise FineTuneBundleError(f"Unsupported delta method {bundle.method!r}.")


def save_finetune_bundle(path: str | Path, bundle: FineTuneBundle) -> FineTuneBundle:
    """Write a ``FineTuneBundle`` to disk and return it."""

    _check_schema(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest, arrays = _bundle_to_manifest(bundle)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        manifest_path = tmp_path / "manifest.json"
        arrays_path = tmp_path / "arrays.eqx"
        with manifest_path.open("w") as handle:
            json.dump(manifest, handle)
        eqx.tree_serialise_leaves(arrays_path, arrays)

        with lz4.frame.open(path, "wb") as archive:
            with tarfile.open(fileobj=archive, mode="w") as tar:
                tar.add(manifest_path, arcname="manifest.json")
                tar.add(arrays_path, arcname="arrays.eqx")
    return bundle


def load_finetune_bundle(
    path: str | Path,
    base_model: PyTree | None = None,
) -> FineTuneBundle | PyTree:
    """Load a bundle, or apply it immediately when ``base_model`` is provided."""

    bundle = _read_bundle(path)
    if base_model is None:
        return bundle
    return load_delta(base_model, bundle)


def merge_and_save(
    path: str | Path,
    model: PyTree,
    *,
    method: str = "lora",
    metadata: dict[str, Any] | None = None,
    model_state: Any | None = None,
    recalibration_required: bool | None = None,
) -> FineTuneBundle:
    """Merge mergeable method weights where safe, then save a delta bundle."""

    if method == "lora":
        return save_delta(
            merge_lora(model),
            path,
            method=method,
            metadata=metadata,
            model_state=model_state,
            recalibration_required=recalibration_required,
        )
    if method == "dora":
        # A merged DoRA model no longer contains DoRA state to serialize as a delta.
        # Save the reversible delta and leave dense export to model serialization.
        return save_delta(
            model,
            path,
            method=method,
            metadata=metadata,
            model_state=model_state,
            recalibration_required=recalibration_required,
        )
    return save_delta(
        model,
        path,
        method=method,
        metadata=metadata,
        model_state=model_state,
        recalibration_required=recalibration_required,
    )


def _read_bundle(path: str | Path) -> FineTuneBundle:
    path = Path(path)
    with lz4.frame.open(path, "rb") as archive:
        with tarfile.open(fileobj=archive, mode="r") as tar:
            manifest_file = tar.extractfile("manifest.json")
            arrays_file = tar.extractfile("arrays.eqx")
            if manifest_file is None or arrays_file is None:
                raise FineTuneBundleError(
                    f"Delta file {path!s} is missing manifest.json or arrays.eqx."
                )
            manifest = json.loads(manifest_file.read().decode())
            arrays_data = io.BytesIO(arrays_file.read())
    bundle = _bundle_from_manifest(manifest, arrays_data)
    _check_schema(bundle)
    return bundle


def _bundle_to_manifest(
    bundle: FineTuneBundle,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays: dict[str, Any] = {}
    payload = {
        "method": bundle.method,
        "schema_version": bundle.schema_version,
        "base_model_name": bundle.base_model_name,
        "base_model_config": bundle.base_model_config,
        "base_checkpoint_id": bundle.base_checkpoint_id,
        "equimo_version": bundle.equimo_version,
        "architecture_hash": bundle.architecture_hash,
        "adapter_config": bundle.adapter_config,
        "selector_spec": bundle.selector_spec,
        "trainable_labels": _labels_to_metadata(bundle.trainable_labels),
        "delta_tree": bundle.delta_tree,
        "model_state": bundle.model_state,
        "lineage": _lineage_to_dict(bundle.lineage),
        "metadata": bundle.metadata,
    }
    return {
        "format": _FORMAT,
        "format_version": _FORMAT_VERSION,
        "bundle": _encode_value(payload, arrays),
    }, arrays


def _bundle_from_manifest(
    manifest: dict[str, Any],
    arrays_data: io.BytesIO,
) -> FineTuneBundle:
    if manifest.get("format") != _FORMAT:
        raise FineTuneBundleError(
            f"Unsupported fine-tuning bundle format {manifest.get('format')!r}."
        )
    if manifest.get("format_version") != _FORMAT_VERSION:
        raise FineTuneBundleError(
            "Unsupported fine-tuning bundle format_version="
            f"{manifest.get('format_version')!r}; expected {_FORMAT_VERSION}."
        )
    encoded = manifest["bundle"]
    template: dict[str, Any] = {}
    _collect_array_templates(encoded, template)
    arrays = eqx.tree_deserialise_leaves(arrays_data, template)
    payload = _decode_value(encoded, arrays)
    if isinstance(payload.get("lineage"), dict):
        payload["lineage"] = _lineage_from_dict(payload["lineage"])
    return FineTuneBundle(**payload)


def _encode_value(value: Any, arrays: dict[str, Any]) -> Any:
    if eqx.is_array(value):
        key = f"array_{len(arrays):06d}"
        array = jnp.asarray(value)
        arrays[key] = value
        return {
            _ARRAY_MARKER: key,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {_TUPLE_MARKER: [_encode_value(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item, arrays) for item in value]
    if isinstance(value, dict):
        encoded: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "FineTuneBundle serialization only supports string dictionary keys; "
                    f"got {key!r}."
                )
            encoded[key] = _encode_value(item, arrays)
        return encoded
    raise TypeError(
        "FineTuneBundle contains a value that cannot be serialized without pickle: "
        f"{type(value).__name__}."
    )


def _collect_array_templates(encoded: Any, template: dict[str, Any]) -> None:
    if isinstance(encoded, dict):
        if _ARRAY_MARKER in encoded:
            template[encoded[_ARRAY_MARKER]] = jax.ShapeDtypeStruct(
                tuple(encoded["shape"]),
                jnp.dtype(encoded["dtype"]),
            )
            return
        for value in encoded.values():
            _collect_array_templates(value, template)
        return
    if isinstance(encoded, list):
        for value in encoded:
            _collect_array_templates(value, template)


def _decode_value(encoded: Any, arrays: dict[str, Any]) -> Any:
    if isinstance(encoded, dict):
        if _ARRAY_MARKER in encoded:
            return arrays[encoded[_ARRAY_MARKER]]
        if _TUPLE_MARKER in encoded:
            return tuple(_decode_value(item, arrays) for item in encoded[_TUPLE_MARKER])
        return {key: _decode_value(value, arrays) for key, value in encoded.items()}
    if isinstance(encoded, list):
        return [_decode_value(item, arrays) for item in encoded]
    return encoded


def _labels_to_metadata(labels: Any) -> dict[str, str]:
    if labels is None:
        return {}
    return {
        path_to_str(key_path_to_path(key_path)): leaf
        for key_path, leaf in jtu.tree_leaves_with_path(labels)
        if isinstance(leaf, str)
    }


def _lineage_to_dict(lineage) -> dict[str, Any]:
    return {
        "base_model_name": lineage.base_model_name,
        "architecture_hash": lineage.architecture_hash,
        "base_checkpoint_id": lineage.base_checkpoint_id,
        "base_checkpoint_hash": lineage.base_checkpoint_hash,
        "base_value_hash": lineage.base_value_hash,
        "preprocessing_fingerprint": lineage.preprocessing_fingerprint,
        "model_state_hash": lineage.model_state_hash,
        "logical_id_table_hash": lineage.logical_id_table_hash,
        "quantization_fingerprint": lineage.quantization_fingerprint,
        "sharding_fingerprint": lineage.sharding_fingerprint,
        "model_revision": lineage.model_revision,
        "parent_bundle_ids": lineage.parent_bundle_ids,
        "identity_stability": lineage.identity_stability,
        "parent_lineages": tuple(
            _lineage_to_dict(item) for item in lineage.parent_lineages
        ),
        "notes": dict(lineage.notes),
    }


def _lineage_from_dict(data: dict[str, Any]):
    from .config import ModelLineage

    parents = tuple(
        _lineage_from_dict(item) if isinstance(item, dict) else item
        for item in data.get("parent_lineages", ())
    )
    return ModelLineage(
        base_model_name=data.get("base_model_name"),
        architecture_hash=data.get("architecture_hash"),
        base_checkpoint_id=data.get("base_checkpoint_id"),
        base_checkpoint_hash=data.get("base_checkpoint_hash"),
        base_value_hash=data.get("base_value_hash"),
        preprocessing_fingerprint=data.get("preprocessing_fingerprint"),
        model_state_hash=data.get("model_state_hash"),
        logical_id_table_hash=data.get("logical_id_table_hash"),
        quantization_fingerprint=data.get("quantization_fingerprint"),
        sharding_fingerprint=data.get("sharding_fingerprint"),
        model_revision=data.get("model_revision"),
        parent_bundle_ids=tuple(data.get("parent_bundle_ids", ())),
        identity_stability=data.get("identity_stability", "path_derived"),
        parent_lineages=parents,
        notes=data.get("notes", {}),
    )


def _extract_prompt_delta(model: PyTree) -> FineTuneBundle:
    if not isinstance(model, PromptedModel):
        raise TypeError("prompt delta saving expects a PromptedModel.")
    return FineTuneBundle(
        method="prompt",
        schema_version=1,
        architecture_hash=architecture_hash(model.base),
        adapter_config={
            "prompts": model.prompts,
            "config": _config_to_dict(model.config),
            "config_class": model.config.__class__.__name__,
        },
    )


def _load_prompt_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    _check_hash(base_model, bundle, "Prompt")
    return PromptedModel(
        base_model,
        tuple(bundle.adapter_config["prompts"]),
        _prompt_config_from_dict(
            bundle.adapter_config["config"],
            bundle.adapter_config.get("config_class", "PromptConfig"),
        ),
    )


def _extract_prefix_delta(model: PyTree) -> FineTuneBundle:
    if not isinstance(model, PrefixTunedModel):
        raise TypeError("prefix delta saving expects a PrefixTunedModel.")
    return FineTuneBundle(
        method="prefix",
        schema_version=1,
        architecture_hash=architecture_hash(strip_prefixes(model.base)),
        adapter_config={
            "prefixes": model.prefixes,
            "prefix_projections": tuple(
                None if projection is None else _prefix_projection_state(projection)
                for projection in model.prefix_projections
            ),
            "config": _config_to_dict(model.config),
        },
    )


def _load_prefix_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    _check_hash(base_model, bundle, "Prefix")
    return PrefixTunedModel(
        base_model,
        tuple(bundle.adapter_config["prefixes"]),
        _prefix_config_from_dict(bundle.adapter_config["config"]),
        prefix_projections=tuple(
            None if state is None else _prefix_projection_from_state(state)
            for state in bundle.adapter_config.get("prefix_projections", ())
        )
        or None,
    )


def _prefix_projection_state(projection: PrefixProjection) -> dict[str, Any]:
    return {
        "down": _linear_state(projection.down),
        "up": _linear_state(projection.up),
        "num_heads": projection.num_heads,
        "head_dim": projection.head_dim,
    }


def _prefix_projection_from_state(state: dict[str, Any]) -> PrefixProjection:
    down = _linear_from_state(state["down"])
    up = _linear_from_state(state["up"])
    return PrefixProjection(
        int(down.in_features),
        int(down.out_features),
        int(state["num_heads"]),
        int(state["head_dim"]),
        key=jr.PRNGKey(0),
        down=down,
        up=up,
    )


def _extract_scale_shift_delta(model: PyTree) -> FineTuneBundle:
    entries = []
    stripped = model
    for path, wrapper in _iter_wrappers(model, ScaleShiftWrapper):
        entries.append(
            {
                "path": path_to_str(path),
                "scale": wrapper.scale_shift.scale,
                "shift": wrapper.scale_shift.shift,
                "axis": wrapper.scale_shift.axis,
                "mergeable": wrapper.mergeable,
            }
        )
        stripped = eqx.tree_at(
            lambda tree, p=path: get_path(tree, p), stripped, wrapper.base
        )
    return FineTuneBundle(
        method="scale_shift",
        schema_version=1,
        architecture_hash=architecture_hash(stripped),
        adapter_config={"entries": entries},
    )


def _load_scale_shift_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    _check_hash(base_model, bundle, "Scale/shift")
    updated = base_model
    for entry in bundle.adapter_config.get("entries", ()):
        path = str_to_path(entry["path"])
        base = _bundle_get_path(updated, path, method_name="Scale/shift")
        dim = int(entry["scale"].shape[0])
        if entry["scale"].shape != entry["shift"].shape:
            raise FineTuneBundleError(
                f"Scale/shift delta expects matching scale and shift shapes at "
                f"{entry['path']}, got {entry['scale'].shape} and {entry['shift'].shape}."
            )
        expected_dim = _scale_shift_dim(base)
        if dim != expected_dim:
            raise FineTuneBundleError(
                f"Scale/shift delta expects path {entry['path']} with feature "
                f"dimension {dim}, got {expected_dim}."
            )
        wrapper = ScaleShiftWrapper(
            base,
            ScaleShift(
                dim,
                axis=entry["axis"],
                scale=entry["scale"],
                shift=entry["shift"],
            ),
            mergeable=bool(entry.get("mergeable", True)),
        )
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, wrapper)
    return updated


def _extract_ia3_delta(model: PyTree) -> FineTuneBundle:
    entries = []
    stripped = model
    for path, wrapper in _iter_wrappers(model, IA3Linear):
        entries.append(
            {
                "path": path_to_str(path),
                "ia3": wrapper.ia3,
                "mergeable": wrapper.mergeable,
                "weight_shape": tuple(wrapper.base.weight.shape),
                "projection_segments": tuple(
                    {
                        "name": segment.name,
                        "axis": segment.axis,
                        "start": segment.start,
                        "stop": segment.stop,
                    }
                    for segment in wrapper.projection_segments
                ),
            }
        )
        stripped = eqx.tree_at(
            lambda tree, p=path: get_path(tree, p), stripped, wrapper.base
        )
    return FineTuneBundle(
        method="ia3",
        schema_version=1,
        architecture_hash=architecture_hash(stripped),
        adapter_config={"entries": entries},
    )


def _load_ia3_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    _check_hash(base_model, bundle, "IA3")
    updated = base_model
    for entry in bundle.adapter_config.get("entries", ()):
        path = str_to_path(entry["path"])
        base = _bundle_get_path(updated, path, method_name="IA3")
        if not isinstance(base, eqx.nn.Linear):
            raise FineTuneBundleError(
                f"IA3 delta expects linear module at {entry['path']}, "
                f"got {type(base).__name__}."
            )
        if tuple(base.weight.shape) != tuple(entry["weight_shape"]):
            raise FineTuneBundleError(
                f"IA3 delta expects path {entry['path']} with shape "
                f"{entry['weight_shape']}, got {tuple(base.weight.shape)}."
            )
        projection_segments = _ia3_projection_segments(entry)
        expected_dim = _ia3_dim(base, projection_segments)
        if tuple(entry["ia3"].shape) != (expected_dim,):
            raise FineTuneBundleError(
                f"IA3 delta expects path {entry['path']} with scale shape "
                f"({expected_dim},), got {tuple(entry['ia3'].shape)}."
            )
        wrapper = IA3Linear(
            base,
            ia3=entry["ia3"],
            projection_segments=projection_segments,
            mergeable=bool(entry.get("mergeable", True)),
        )
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, wrapper)
    return updated


def _extract_vera_delta(model: PyTree) -> FineTuneBundle:
    entries = []
    for path, wrapper in _iter_wrappers(model, VeRALinear):
        entries.append(
            {
                "path": path_to_str(path),
                "rank": int(wrapper.vera_A.shape[0]),
                "shared": wrapper.shared,
                "frozen_A_init": wrapper.frozen_A_init,
                "frozen_B_init": wrapper.frozen_B_init,
                "mergeable": wrapper.mergeable,
                "basis_generation": wrapper.basis_generation,
                "basis_key_data": wrapper.basis_key_data,
                "basis_pool_key": wrapper.basis_pool_key,
                "share_scope": wrapper.share_scope,
                "input_scale_axis": "rank",
                "output_scale_axis": "logical_output",
                "base_weight_shape": tuple(wrapper.base.weight.shape),
                "base_bias_shape": None
                if wrapper.base.bias is None
                else tuple(wrapper.base.bias.shape),
                "vera_A": wrapper.vera_A,
                "vera_B": wrapper.vera_B,
                "vera_input_scale": wrapper.vera_input_scale,
                "vera_output_scale": wrapper.vera_output_scale,
            }
        )
    return FineTuneBundle(
        method="vera",
        schema_version=1,
        architecture_hash=architecture_hash(strip_vera(model)),
        adapter_config={"entries": entries},
    )


def _load_vera_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    _check_hash(base_model, bundle, "VeRA")
    updated = base_model
    for entry in bundle.adapter_config.get("entries", ()):
        path = str_to_path(entry["path"])
        base = _bundle_get_path(updated, path, method_name="VeRA")
        if not isinstance(base, eqx.nn.Linear):
            raise FineTuneBundleError(
                f"VeRA delta expects linear module at {entry['path']}, "
                f"got {type(base).__name__}."
            )
        if tuple(base.weight.shape) != tuple(entry["base_weight_shape"]):
            raise FineTuneBundleError(
                f"VeRA delta expects path {entry['path']} with weight shape "
                f"{entry['base_weight_shape']}, got {tuple(base.weight.shape)}."
            )
        expected_bias_shape = entry["base_bias_shape"]
        actual_bias_shape = None if base.bias is None else tuple(base.bias.shape)
        if actual_bias_shape != expected_bias_shape:
            raise FineTuneBundleError(
                f"VeRA delta expects path {entry['path']} with bias shape "
                f"{expected_bias_shape}, got {actual_bias_shape}."
            )
        wrapper = VeRALinear(
            base,
            rank=int(entry["rank"]),
            key=jr.PRNGKey(0),
            shared=bool(entry.get("shared", True)),
            frozen_A_init=entry.get("frozen_A_init", "kaiming_uniform"),
            frozen_B_init=entry.get("frozen_B_init", "kaiming_uniform"),
            mergeable=bool(entry.get("mergeable", True)),
            vera_A=entry["vera_A"],
            vera_B=entry["vera_B"],
            vera_input_scale=entry["vera_input_scale"],
            vera_output_scale=entry["vera_output_scale"],
            basis_generation=entry.get("basis_generation", "unknown_legacy"),
            basis_key_data=tuple(entry.get("basis_key_data", ())),
            basis_pool_key=tuple(entry.get("basis_pool_key", ())),
            share_scope=entry.get(
                "share_scope",
                "shape_compatible" if bool(entry.get("shared", True)) else "per_module",
            ),
        )
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, wrapper)
    return updated


def _linear_state(linear: eqx.nn.Linear) -> dict[str, Any]:
    return {
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "use_bias": linear.bias is not None,
        "weight": linear.weight,
        "bias": linear.bias,
    }


def _linear_from_state(state: dict[str, Any]) -> eqx.nn.Linear:
    linear = eqx.nn.Linear(
        int(state["in_features"]),
        int(state["out_features"]),
        use_bias=bool(state["use_bias"]),
        key=jr.PRNGKey(0),
    )
    linear = eqx.tree_at(lambda layer: layer.weight, linear, state["weight"])
    if state["bias"] is not None:
        linear = eqx.tree_at(lambda layer: layer.bias, linear, state["bias"])
    return linear


def _extract_dora_delta(model: PyTree) -> FineTuneBundle:
    entries = []
    stripped = model
    for path, wrapper in _iter_wrappers(model, DoRALinear):
        entries.append(
            {
                "path": path_to_str(path),
                "class": wrapper.__class__.__name__,
                "rank": wrapper.rank,
                "alpha": wrapper.alpha,
                "scaling": wrapper.scaling_mode,
                "dropout": wrapper.dropout,
                "eps": wrapper.eps,
                "norm_gradient": wrapper.norm_gradient,
                "norm_impl": wrapper.norm_impl,
                "train_base": wrapper.train_base,
                "mergeable": wrapper.mergeable,
                "lora_A": wrapper.lora_A,
                "lora_B": wrapper.lora_B,
                "magnitude": wrapper.magnitude,
                "weight_shape": tuple(wrapper.base.weight.shape),
            }
        )
        stripped = eqx.tree_at(
            lambda tree, p=path: get_path(tree, p), stripped, wrapper.base
        )
    return FineTuneBundle(
        method="dora",
        schema_version=1,
        architecture_hash=architecture_hash(stripped),
        adapter_config={"entries": entries},
    )


def _load_dora_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    _check_hash(base_model, bundle, "DoRA")
    updated = base_model
    for entry in bundle.adapter_config.get("entries", ()):
        path = str_to_path(entry["path"])
        base = _bundle_get_path(updated, path, method_name="DoRA")
        if not isinstance(base, eqx.nn.Linear):
            raise FineTuneBundleError(
                f"DoRA delta expects linear module at {entry['path']}, "
                f"got {type(base).__name__}."
            )
        if tuple(base.weight.shape) != tuple(entry["weight_shape"]):
            raise FineTuneBundleError(
                f"DoRA delta expects path {entry['path']} with shape "
                f"{entry['weight_shape']}, got {tuple(base.weight.shape)}."
            )
        wrapper_type = (
            DoRAMergedLinear if entry["class"] == "DoRAMergedLinear" else DoRALinear
        )
        wrapper = wrapper_type(
            base,
            rank=int(entry["rank"]),
            alpha=float(entry["alpha"]),
            scaling=entry["scaling"],
            dropout=float(entry.get("dropout", 0.0)),
            eps=float(entry["eps"]),
            norm_gradient=entry.get("norm_gradient", "full"),
            norm_impl=entry.get("norm_impl", "dense"),
            train_base=bool(entry.get("train_base", False)),
            mergeable=bool(entry.get("mergeable", True)),
            key=jr.PRNGKey(0),
            lora_A=entry["lora_A"],
            lora_B=entry["lora_B"],
            magnitude=entry["magnitude"],
        )
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, wrapper)
    return updated


def _check_hash(base_model: PyTree, bundle: FineTuneBundle, method_name: str) -> None:
    actual_hash = architecture_hash(base_model)
    if bundle.architecture_hash and bundle.architecture_hash != actual_hash:
        raise FineTuneBundleError(
            f"{method_name} delta architecture hash mismatch: "
            f"expected {bundle.architecture_hash}, got {actual_hash}."
        )


def _check_schema(bundle: FineTuneBundle) -> None:
    if bundle.schema_version != 1:
        raise FineTuneBundleError(
            f"Unsupported fine-tuning bundle schema_version={bundle.schema_version!r}; "
            "expected 1."
        )


def _bundle_get_path(model: PyTree, path: tuple[str | int, ...], *, method_name: str):
    try:
        return get_path(model, path)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise FineTuneBundleError(
            f"{method_name} delta expects path {path_to_str(path)}, "
            "but the base model has no matching leaf."
        ) from error


def _scale_shift_dim(module: eqx.Module) -> int:
    if isinstance(module, eqx.nn.Linear):
        return int(module.out_features)
    if hasattr(module, "shape"):
        shape = module.shape
        return int(shape[0] if isinstance(shape, tuple) else shape)
    if hasattr(module, "weight"):
        return int(module.weight.shape[0])
    raise FineTuneBundleError(
        f"Scale/shift delta cannot infer target dimension for {type(module).__name__}."
    )


def _ia3_projection_segments(entry: dict[str, Any]) -> tuple[ProjectionSegment, ...]:
    return tuple(
        ProjectionSegment(
            name=item["name"],
            axis=int(item["axis"]),
            start=int(item["start"]),
            stop=int(item["stop"]),
        )
        for item in entry.get("projection_segments", ())
    )


def _ia3_dim(
    base: eqx.nn.Linear,
    projection_segments: tuple[ProjectionSegment, ...],
) -> int:
    if not projection_segments:
        return int(base.out_features)
    return sum(segment.stop - segment.start for segment in projection_segments)


def _iter_wrappers(model: PyTree, wrapper_type: type):
    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, wrapper_type),
        )
        if isinstance(leaf, wrapper_type)
    )


def _resolve_save_delta_args(
    args,
    *,
    model: PyTree | None,
    path: str | Path | None,
    base_model: PyTree | None,
    spec: Any | None,
) -> tuple[PyTree, str | Path, PyTree | None, Any | None]:
    resolved_model = model
    resolved_path = path
    resolved_base_model = base_model
    resolved_spec = spec

    if len(args) > 4:
        raise TypeError("save_delta accepts at most four positional arguments.")

    if args:
        if _is_pathlike(args[0]):
            if resolved_path is not None:
                raise TypeError(
                    "save_delta path was provided both positionally and by keyword."
                )
            resolved_path = args[0]
            if len(args) >= 2:
                if resolved_model is not None:
                    raise TypeError(
                        "save_delta model was provided both positionally and by keyword."
                    )
                resolved_model = args[1]
            if len(args) >= 3:
                if resolved_base_model is not None:
                    raise TypeError(
                        "save_delta base_model was provided both positionally and by keyword."
                    )
                resolved_base_model = args[2]
            if len(args) >= 4:
                if resolved_spec is not None:
                    raise TypeError(
                        "save_delta spec was provided both positionally and by keyword."
                    )
                resolved_spec = args[3]
        else:
            if resolved_model is not None:
                raise TypeError(
                    "save_delta model was provided both positionally and by keyword."
                )
            resolved_model = args[0]
            if len(args) >= 2:
                if resolved_path is not None:
                    raise TypeError(
                        "save_delta path was provided both positionally and by keyword."
                    )
                resolved_path = args[1]
            if len(args) >= 3:
                if resolved_base_model is not None:
                    raise TypeError(
                        "save_delta base_model was provided both positionally and by keyword."
                    )
                resolved_base_model = args[2]
            if len(args) >= 4:
                if resolved_spec is not None:
                    raise TypeError(
                        "save_delta spec was provided both positionally and by keyword."
                    )
                resolved_spec = args[3]

    if resolved_model is None or resolved_path is None:
        raise TypeError("save_delta requires a model and path.")
    if not _is_pathlike(resolved_path):
        raise TypeError("save_delta path must be a str or pathlib.Path.")
    return resolved_model, resolved_path, resolved_base_model, resolved_spec


def _resolve_load_delta_args(args) -> tuple[PyTree, str | Path | FineTuneBundle]:
    if len(args) != 2:
        raise TypeError("load_delta requires a base model and path_or_bundle.")
    first, second = args
    if _is_pathlike(first) or isinstance(first, FineTuneBundle):
        return second, first
    return first, second


def _is_pathlike(value: Any) -> bool:
    return isinstance(value, (str, Path))


def _enrich_bundle(
    bundle: FineTuneBundle,
    model: PyTree,
    *,
    base_model: PyTree | None,
    spec: Any | None,
    user_metadata: dict[str, Any] | None,
    model_state: Any | None = None,
    recalibration_required: bool | None = None,
) -> FineTuneBundle:
    metadata_model = (
        base_model
        if base_model is not None
        else _metadata_base_model(model, bundle.method)
    )
    stats = _bundle_stats(model, metadata_model, bundle)
    if spec is not None:
        stats["spec"] = _metadata_repr(spec)
    if user_metadata:
        stats["user_metadata"] = user_metadata
    selector_spec = (
        bundle.selector_spec or _selector_spec_from_spec(spec) or _selector_spec(bundle)
    )
    base_checkpoint_id = bundle.base_checkpoint_id
    if base_checkpoint_id is None and (
        base_model is not None or _can_infer_exact_base_checkpoint(model, bundle.method)
    ):
        base_checkpoint_id = _checkpoint_hash(metadata_model)
    base_value_hash = (
        base_checkpoint_id if base_checkpoint_id else bundle.lineage.base_value_hash
    )
    logical_id_hash = _logical_id_table_hash(bundle)
    quantization_fingerprint = _bundle_quantization_fingerprint(bundle)
    calibration_refs = _bundle_calibration_references(bundle)
    model_state_snapshot = bundle.model_state if model_state is None else model_state
    model_state_hash = _model_state_hash(model_state_snapshot)
    recalibration_marker = (
        bool(recalibration_required)
        if recalibration_required is not None
        else bool(
            bundle.metadata.get(
                "recalibration_required",
                False if model_state_snapshot is not None else False,
            )
        )
    )
    stats.setdefault(
        "method_profile_id", bundle.metadata.get("method_profile_id", bundle.method)
    )
    stats.setdefault(
        "primary_references", bundle.metadata.get("primary_references", ())
    )
    stats.setdefault("calibration_artifact_references", calibration_refs)
    stats.setdefault("model_state_present", model_state_snapshot is not None)
    stats.setdefault("model_state_hash", model_state_hash)
    stats.setdefault("recalibration_required", recalibration_marker)
    stats.setdefault("quantization_fingerprint", quantization_fingerprint)
    stats.setdefault("equimo_source_revision", _equimo_version() or "unknown")
    lineage = replace(
        bundle.lineage if isinstance(bundle.lineage, ModelLineage) else ModelLineage(),
        base_model_name=bundle.lineage.base_model_name or _model_name(metadata_model),
        architecture_hash=bundle.lineage.architecture_hash or bundle.architecture_hash,
        base_checkpoint_id=bundle.lineage.base_checkpoint_id or base_checkpoint_id,
        base_checkpoint_hash=bundle.lineage.base_checkpoint_hash or base_value_hash,
        base_value_hash=base_value_hash,
        model_state_hash=bundle.lineage.model_state_hash or model_state_hash,
        logical_id_table_hash=logical_id_hash,
        quantization_fingerprint=quantization_fingerprint,
        model_revision=bundle.lineage.model_revision or _equimo_version() or None,
    )
    return replace(
        bundle,
        base_checkpoint_id=base_checkpoint_id,
        base_model_name=bundle.base_model_name or _model_name(metadata_model),
        base_model_config=bundle.base_model_config or _model_config(metadata_model),
        equimo_version=bundle.equimo_version or _equimo_version(),
        selector_spec=selector_spec,
        lineage=lineage,
        trainable_labels=bundle.trainable_labels
        if bundle.trainable_labels is not None
        else _trainable_labels(model, bundle.method),
        model_state=model_state_snapshot,
        metadata={**stats, **dict(bundle.metadata)},
    )


def _bundle_stats(
    model: PyTree, base_model: PyTree, bundle: FineTuneBundle
) -> dict[str, Any]:
    total_params = _param_count(base_model)
    trainable_params = _delta_param_count(bundle)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_fraction": trainable_params / total_params if total_params else 0.0,
        "dtype_summary": _dtype_summary(model),
        "target_paths": _target_paths(bundle),
        "mergeable": _is_mergeable(bundle),
        "merged": _is_merged(bundle),
    }


def _logical_id_table_hash(bundle: FineTuneBundle) -> str:
    entries = tuple(
        {
            "path": str(entry.get("path", "")),
            "class": str(entry.get("class", "")),
            "projection_segments": entry.get("projection_segments", ()),
            "factor_convention": entry.get("factor_convention"),
        }
        for entry in bundle.adapter_config.get("entries", ())
    )
    if not entries:
        entries = tuple({"path": path} for path in _target_paths(bundle))
    return _hash_json(entries)


def _bundle_quantization_fingerprint(bundle: FineTuneBundle) -> str | None:
    records = []
    for entry in bundle.adapter_config.get("entries", ()):
        metadata = _entry_metadata(entry)
        fingerprint = metadata.get("quantization_fingerprint")
        if fingerprint:
            records.append(
                {
                    "path": str(entry.get("path", "")),
                    "fingerprint": fingerprint,
                }
            )
    if not records:
        return None
    return _hash_json(records)


def _bundle_calibration_references(
    bundle: FineTuneBundle,
) -> tuple[dict[str, str], ...]:
    references = []
    for entry in bundle.adapter_config.get("entries", ()):
        metadata = _entry_metadata(entry)
        if not any(key.startswith("calibration_") for key in metadata):
            continue
        references.append(
            {
                "path": str(entry.get("path", "")),
                "kind": metadata.get("calibration_kind", ""),
                "sample_count": metadata.get("calibration_sample_count", ""),
                "data_fingerprint": metadata.get("calibration_data_fingerprint", ""),
                "reduction": metadata.get("calibration_reduction", ""),
                "base_checkpoint_hash": metadata.get(
                    "calibration_base_checkpoint_hash", ""
                ),
            }
        )
    return tuple(references)


def _entry_metadata(entry: dict[str, Any]) -> dict[str, str]:
    metadata = entry.get("metadata", ())
    if isinstance(metadata, dict):
        return {str(key): str(value) for key, value in metadata.items()}
    return {str(key): str(value) for key, value in tuple(metadata)}


def _model_state_hash(model_state: Any | None) -> str | None:
    if model_state is None:
        return None
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if eqx.is_array(value):
            array = np.asarray(value)
            digest.update(b"array")
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
            return
        if value is None or isinstance(value, (bool, int, float, str)):
            digest.update(json.dumps(value, sort_keys=True).encode())
            return
        if isinstance(value, tuple):
            digest.update(b"tuple")
            for item in value:
                update(item)
            return
        if isinstance(value, list):
            digest.update(b"list")
            for item in value:
                update(item)
            return
        if isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value):
                if not isinstance(key, str):
                    raise TypeError("model_state dictionary keys must be strings.")
                digest.update(key.encode())
                update(value[key])
            return
        raise TypeError(
            "model_state must be bundle-serializable: arrays, scalars, "
            f"tuples, lists, and string-keyed dicts; got {type(value).__name__}."
        )

    update(model_state)
    return f"sha256:{digest.hexdigest()}"


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _param_count(tree: PyTree) -> int:
    return int(
        sum(
            leaf.size
            for leaf in jtu.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
            if eqx.is_inexact_array(leaf)
        )
    )


def _checkpoint_hash(tree: PyTree) -> str:
    digest = hashlib.sha256()
    for key_path, leaf in jtu.tree_leaves_with_path(
        eqx.filter(tree, eqx.is_inexact_array)
    ):
        if not eqx.is_inexact_array(leaf):
            continue
        array = np.asarray(leaf)
        digest.update(path_to_str(key_path_to_path(key_path)).encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _check_base_checkpoint(base_model: PyTree, bundle: FineTuneBundle) -> None:
    expected = bundle.base_checkpoint_id or bundle.lineage.base_value_hash
    if not expected or not expected.startswith("sha256:"):
        return
    if (
        bundle.architecture_hash
        and architecture_hash(base_model) != bundle.architecture_hash
    ):
        return
    actual = _checkpoint_hash(base_model)
    if actual != expected:
        raise FineTuneBundleError(
            f"{bundle.method} delta base checkpoint mismatch: "
            f"expected {expected}, got {actual}."
        )


def _check_bundle_lineage_consistency(bundle: FineTuneBundle) -> None:
    expected_logical = bundle.lineage.logical_id_table_hash
    actual_logical = _logical_id_table_hash(bundle)
    if expected_logical and expected_logical != actual_logical:
        raise FineTuneBundleError(
            f"{bundle.method} delta logical-ID table mismatch: "
            f"expected {expected_logical}, got {actual_logical}."
        )
    expected_quantization = bundle.lineage.quantization_fingerprint
    actual_quantization = _bundle_quantization_fingerprint(bundle)
    if expected_quantization != actual_quantization:
        raise FineTuneBundleError(
            f"{bundle.method} delta quantization fingerprint mismatch: "
            f"expected {expected_quantization}, got {actual_quantization}."
        )


def _metadata_base_model(model: PyTree, method: str) -> PyTree:
    if method == "lora":
        return strip_lora(model)
    if method == "adapter":
        return strip_adapters(model)
    if method == "prompt" and isinstance(model, PromptedModel):
        return model.base
    if method == "prefix" and isinstance(model, PrefixTunedModel):
        return strip_prefixes(model.base)
    if method == "scale_shift":
        return _strip_wrappers(model, ScaleShiftWrapper)
    if method == "ia3":
        return _strip_wrappers(model, IA3Linear)
    if method == "vera":
        return strip_vera(model)
    if method == "dora":
        return _strip_wrappers(model, DoRALinear)
    return model


def _can_infer_exact_base_checkpoint(model: PyTree, method: str) -> bool:
    if method != "lora":
        return True
    return all(
        module.base_weight_delta is None for _, module in iter_lora_modules(model)
    )


def _strip_wrappers(model: PyTree, wrapper_type: type) -> PyTree:
    stripped = model
    for path, wrapper in _iter_wrappers(stripped, wrapper_type):
        base = getattr(wrapper, "base", None)
        if base is None:
            continue
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, base)
    return stripped


def _delta_param_count(bundle: FineTuneBundle) -> int:
    if bundle.method == "vera":
        return int(
            sum(
                entry["vera_input_scale"].size + entry["vera_output_scale"].size
                for entry in bundle.adapter_config.get("entries", ())
            )
        )
    leaves = jtu.tree_leaves(bundle.adapter_config)
    return int(sum(leaf.size for leaf in leaves if eqx.is_inexact_array(leaf)))


def _dtype_summary(tree: PyTree) -> dict[str, int]:
    summary: dict[str, int] = {}
    for leaf in jtu.tree_leaves(eqx.filter(tree, eqx.is_inexact_array)):
        if not eqx.is_inexact_array(leaf):
            continue
        dtype = str(jnp.asarray(leaf).dtype)
        summary[dtype] = summary.get(dtype, 0) + int(leaf.size)
    return summary


def _target_paths(bundle: FineTuneBundle) -> tuple[str, ...]:
    entries = bundle.adapter_config.get("entries", ())
    if entries:
        return tuple(str(entry["path"]) for entry in entries if "path" in entry)
    if "prompts" in bundle.adapter_config:
        return ("prompts",)
    if "prefixes" in bundle.adapter_config:
        return ("prefixes",)
    return ()


def _selector_spec(bundle: FineTuneBundle) -> dict[str, Any]:
    paths = _target_paths(bundle)
    return {"paths": paths} if paths else {}


def _selector_spec_from_spec(spec: Any | None) -> dict[str, Any]:
    if spec is None:
        return {}
    if isinstance(spec, TargetSpec):
        return {"target": _target_spec_to_dict(spec)}
    target = getattr(spec, "target", None)
    if isinstance(target, TargetSpec):
        return {"target": _target_spec_to_dict(target)}
    return {}


def _target_spec_to_dict(target: TargetSpec) -> dict[str, Any]:
    return {
        "tags_all": target.tags_all,
        "tags_any": target.tags_any,
        "include": target.include,
        "exclude": target.exclude,
        "min_depth": target.min_depth,
        "max_depth": target.max_depth,
        "target_kind": target.target_kind,
        "allow_empty": target.allow_empty,
        "predicate": None
        if target.predicate is None
        else getattr(target.predicate, "__name__", "<callable>"),
    }


def _config_to_dict(config: Any) -> dict[str, Any]:
    return {
        key: value for key, value in vars(config).items() if not key.startswith("_")
    }


def _prompt_config_from_dict(
    config: dict[str, Any],
    class_name: str = "PromptConfig",
) -> PromptConfig:
    config_type = {
        "PromptConfig": PromptConfig,
        "SoftPromptConfig": SoftPromptConfig,
        "PTuningV2Config": PTuningV2Config,
        "VPTDeepConfig": VPTDeepConfig,
        "VPTShallowConfig": VPTShallowConfig,
    }.get(class_name, PromptConfig)
    return config_type(**config)


def _prefix_config_from_dict(config: dict[str, Any]) -> PrefixConfig:
    return PrefixConfig(**config)


def _trainable_labels(model: PyTree, method: str):
    from .config import TrainableSpec
    from .surgery import prepare_finetune

    plan = prepare_finetune(
        model,
        trainable=TrainableSpec(
            mode="peft",
            method_name=method,
            train_head=False,
        ),
    )
    return plan.labels


def _model_name(model: PyTree) -> str:
    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__qualname__}"


def _model_config(model: PyTree) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name, value in vars(model).items():
        if name.startswith("_"):
            continue
        metadata_value = _metadata_scalar(value)
        if metadata_value is not _UNSERIALIZABLE:
            config[name] = metadata_value
    return config


_UNSERIALIZABLE = object()


def _metadata_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        values = tuple(_metadata_scalar(item) for item in value)
        if any(item is _UNSERIALIZABLE for item in values):
            return _UNSERIALIZABLE
        return values
    if isinstance(value, list):
        values = [_metadata_scalar(item) for item in value]
        if any(item is _UNSERIALIZABLE for item in values):
            return _UNSERIALIZABLE
        return values
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            metadata_item = _metadata_scalar(item)
            if metadata_item is _UNSERIALIZABLE:
                return _UNSERIALIZABLE
            result[key] = metadata_item
        return result
    return _UNSERIALIZABLE


def _is_mergeable(bundle: FineTuneBundle) -> bool:
    if bundle.method in {"ia3", "lora", "scale_shift"}:
        entries = bundle.adapter_config.get("entries", ())
        return all(bool(entry.get("mergeable", True)) for entry in entries)
    if bundle.method == "vera":
        entries = bundle.adapter_config.get("entries", ())
        return all(bool(entry.get("mergeable", True)) for entry in entries)
    return bundle.method == "dora"


def _is_merged(bundle: FineTuneBundle) -> bool:
    entries = bundle.adapter_config.get("entries", ())
    return any(bool(entry.get("merged", False)) for entry in entries)


def _metadata_repr(value: Any) -> Any:
    if callable(value):
        return getattr(value, "__name__", "<callable>")
    if hasattr(value, "__dict__"):
        return {
            key: _metadata_repr(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    if isinstance(value, (tuple, list)):
        return tuple(_metadata_repr(item) for item in value)
    if isinstance(value, dict):
        return {key: _metadata_repr(item) for key, item in value.items()}
    return value


def _equimo_version() -> str:
    for name in ("Equimo", "equimo"):
        try:
            return package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            continue
    return ""


__all__ = (
    "load_delta",
    "load_finetune_bundle",
    "merge_and_save",
    "save_delta",
    "save_finetune_bundle",
)
