"""Fine-tuning delta serialization."""

from __future__ import annotations

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

from ._typing import PyTree
from .config import FineTuneBundle, TargetSpec
from .paths import key_path_to_path, path_to_str, str_to_path
from .peft.adapters import extract_adapter_delta, load_adapter_delta
from .peft.base import get_path
from .peft.dora import DoRALinear, DoRAMergedLinear
from .peft.ia3 import IA3Linear
from .peft.lora import extract_lora_delta, load_lora_delta
from .peft.lora import architecture_hash, merge_lora
from .peft.prefix import PrefixConfig, PrefixTunedModel, strip_prefixes
from .peft.prompts import PromptConfig, PromptedModel
from .peft.scale_shift import ScaleShift, ScaleShiftWrapper

_FORMAT = "equimo.finetune.bundle"
_FORMAT_VERSION = 1
_ARRAY_MARKER = "__equimo_finetune_array__"
_TUPLE_MARKER = "__equimo_finetune_tuple__"


def save_delta(
    *args,
    method: str = "lora",
    metadata: dict[str, Any] | None = None,
    base_model: PyTree | None = None,
    spec: Any | None = None,
) -> FineTuneBundle:
    """Save a method delta bundle and return the saved bundle.

    Both ``save_delta(model, path, ...)`` and the spec-style
    ``save_delta(path, model, base_model, spec)`` call order are accepted.
    ``base_model`` and ``spec`` are metadata inputs; optimizers remain external.
    """

    model, path, resolved_base_model, resolved_spec = _resolve_save_delta_args(
        args,
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
    else:
        raise ValueError(
            "Unsupported delta method "
            f"{method!r}; currently 'lora', 'dora', 'adapter', 'prompt', "
            "'prefix', 'scale_shift', or 'ia3'."
        )

    bundle = _enrich_bundle(
        bundle,
        model,
        base_model=resolved_base_model,
        spec=resolved_spec,
        user_metadata=metadata,
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
    raise ValueError(f"Unsupported delta method {bundle.method!r}.")


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
) -> FineTuneBundle:
    """Merge mergeable method weights where safe, then save a delta bundle."""

    if method == "lora":
        return save_delta(merge_lora(model), path, method=method, metadata=metadata)
    if method == "dora":
        # A merged DoRA model no longer contains DoRA state to serialize as a delta.
        # Save the reversible delta and leave dense export to model serialization.
        return save_delta(model, path, method=method, metadata=metadata)
    return save_delta(model, path, method=method, metadata=metadata)


def _read_bundle(path: str | Path) -> FineTuneBundle:
    path = Path(path)
    with lz4.frame.open(path, "rb") as archive:
        with tarfile.open(fileobj=archive, mode="r") as tar:
            manifest_file = tar.extractfile("manifest.json")
            arrays_file = tar.extractfile("arrays.eqx")
            if manifest_file is None or arrays_file is None:
                raise ValueError(
                    f"Delta file {path!s} is missing manifest.json or arrays.eqx."
                )
            manifest = json.loads(manifest_file.read().decode())
            arrays_data = io.BytesIO(arrays_file.read())
    bundle = _bundle_from_manifest(manifest, arrays_data)
    _check_schema(bundle)
    return bundle


def _bundle_to_manifest(bundle: FineTuneBundle) -> tuple[dict[str, Any], dict[str, Any]]:
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
        raise ValueError(
            f"Unsupported fine-tuning bundle format {manifest.get('format')!r}."
        )
    if manifest.get("format_version") != _FORMAT_VERSION:
        raise ValueError(
            "Unsupported fine-tuning bundle format_version="
            f"{manifest.get('format_version')!r}; expected {_FORMAT_VERSION}."
        )
    encoded = manifest["bundle"]
    template: dict[str, Any] = {}
    _collect_array_templates(encoded, template)
    arrays = eqx.tree_deserialise_leaves(arrays_data, template)
    payload = _decode_value(encoded, arrays)
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


def _extract_prompt_delta(model: PyTree) -> FineTuneBundle:
    if not isinstance(model, PromptedModel):
        raise TypeError("prompt delta saving expects a PromptedModel.")
    return FineTuneBundle(
        method="prompt",
        schema_version=1,
        architecture_hash=architecture_hash(model.base),
        adapter_config={"prompts": model.prompts, "config": _config_to_dict(model.config)},
    )


def _load_prompt_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    _check_hash(base_model, bundle, "Prompt")
    return PromptedModel(
        base_model,
        tuple(bundle.adapter_config["prompts"]),
        _prompt_config_from_dict(bundle.adapter_config["config"]),
    )


def _extract_prefix_delta(model: PyTree) -> FineTuneBundle:
    if not isinstance(model, PrefixTunedModel):
        raise TypeError("prefix delta saving expects a PrefixTunedModel.")
    return FineTuneBundle(
        method="prefix",
        schema_version=1,
        architecture_hash=architecture_hash(strip_prefixes(model.base)),
        adapter_config={"prefixes": model.prefixes, "config": _config_to_dict(model.config)},
    )


def _load_prefix_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    _check_hash(base_model, bundle, "Prefix")
    return PrefixTunedModel(
        base_model,
        tuple(bundle.adapter_config["prefixes"]),
        _prefix_config_from_dict(bundle.adapter_config["config"]),
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
            }
        )
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, wrapper.base)
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
        base = get_path(updated, path)
        dim = int(entry["scale"].shape[0])
        wrapper = ScaleShiftWrapper(
            base,
            ScaleShift(
                dim,
                axis=entry["axis"],
                scale=entry["scale"],
                shift=entry["shift"],
            ),
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
                "weight_shape": tuple(wrapper.base.weight.shape),
            }
        )
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, wrapper.base)
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
        base = get_path(updated, path)
        if tuple(base.weight.shape) != tuple(entry["weight_shape"]):
            raise ValueError(
                f"IA3 delta expects path {entry['path']} with shape "
                f"{entry['weight_shape']}, got {tuple(base.weight.shape)}."
            )
        wrapper = IA3Linear(base, ia3=entry["ia3"])
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, wrapper)
    return updated


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
                "eps": wrapper.eps,
                "lora_A": wrapper.lora_A,
                "lora_B": wrapper.lora_B,
                "magnitude": wrapper.magnitude,
                "weight_shape": tuple(wrapper.base.weight.shape),
            }
        )
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, wrapper.base)
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
        base = get_path(updated, path)
        if tuple(base.weight.shape) != tuple(entry["weight_shape"]):
            raise ValueError(
                f"DoRA delta expects path {entry['path']} with shape "
                f"{entry['weight_shape']}, got {tuple(base.weight.shape)}."
            )
        wrapper_type = DoRAMergedLinear if entry["class"] == "DoRAMergedLinear" else DoRALinear
        wrapper = wrapper_type(
            base,
            rank=int(entry["rank"]),
            alpha=float(entry["alpha"]),
            scaling=entry["scaling"],
            eps=float(entry["eps"]),
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
        raise ValueError(
            f"{method_name} delta architecture hash mismatch: "
            f"expected {bundle.architecture_hash}, got {actual_hash}."
        )


def _check_schema(bundle: FineTuneBundle) -> None:
    if bundle.schema_version != 1:
        raise ValueError(
            f"Unsupported fine-tuning bundle schema_version={bundle.schema_version!r}; "
            "expected 1."
        )


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
    base_model: PyTree | None,
    spec: Any | None,
) -> tuple[PyTree, str | Path, PyTree | None, Any | None]:
    if len(args) < 2:
        raise TypeError("save_delta requires a model and path.")

    if _is_pathlike(args[0]):
        path = args[0]
        model = args[1]
        resolved_base_model = args[2] if len(args) >= 3 else base_model
        resolved_spec = args[3] if len(args) >= 4 else spec
    else:
        model = args[0]
        path = args[1]
        resolved_base_model = args[2] if len(args) >= 3 else base_model
        resolved_spec = args[3] if len(args) >= 4 else spec

    if len(args) > 4:
        raise TypeError("save_delta accepts at most four positional arguments.")
    if not _is_pathlike(path):
        raise TypeError("save_delta path must be a str or pathlib.Path.")
    return model, path, resolved_base_model, resolved_spec


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
) -> FineTuneBundle:
    stats_model = base_model if base_model is not None else model
    stats = _bundle_stats(model, stats_model, bundle)
    if spec is not None:
        stats["spec"] = _metadata_repr(spec)
    if user_metadata:
        stats["user_metadata"] = user_metadata
    metadata_model = base_model if base_model is not None else model
    selector_spec = bundle.selector_spec or _selector_spec_from_spec(spec) or _selector_spec(bundle)
    return replace(
        bundle,
        base_model_name=bundle.base_model_name or _model_name(metadata_model),
        base_model_config=bundle.base_model_config or _model_config(metadata_model),
        equimo_version=bundle.equimo_version or _equimo_version(),
        selector_spec=selector_spec,
        trainable_labels=bundle.trainable_labels
        if bundle.trainable_labels is not None
        else _trainable_labels(model, bundle.method),
        metadata={**stats, **dict(bundle.metadata)},
    )


def _bundle_stats(model: PyTree, base_model: PyTree, bundle: FineTuneBundle) -> dict[str, Any]:
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


def _param_count(tree: PyTree) -> int:
    return int(
        sum(
            leaf.size
            for leaf in jtu.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
            if eqx.is_inexact_array(leaf)
        )
    )


def _delta_param_count(bundle: FineTuneBundle) -> int:
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
        "include": target.include,
        "exclude": target.exclude,
        "tags": target.tags,
        "min_depth": target.min_depth,
        "max_depth": target.max_depth,
        "predicate": None
        if target.predicate is None
        else getattr(target.predicate, "__name__", "<callable>"),
    }


def _config_to_dict(config: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_")
    }


def _prompt_config_from_dict(config: dict[str, Any]) -> PromptConfig:
    return PromptConfig(**config)


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
    if bundle.method == "lora":
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
