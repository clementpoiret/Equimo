"""LoRA modules and model surgery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..config import (
    FineTuneBundle,
    FineTuneBundleError,
    ProjectionSegment,
    TargetSpec,
    WeightLayout,
)
from ..paths import key_path_to_path, path_to_str, str_to_path
from ..selectors import resolve_target
from ..tags import Tagger, canonical_tags_for_path
from .base import get_path


ScalingMode = Literal["alpha_over_r", "alpha_over_sqrt_r"]


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for applying LoRA to linear modules."""

    rank: int = 8
    alpha: float = 16.0
    scaling: ScalingMode = "alpha_over_r"
    dropout: float = 0.0
    target: TargetSpec = field(
        default_factory=lambda: TargetSpec(
            tags_any=("attention.qkv", "attention.proj"),
        )
    )
    init: str = "kaiming_A_zero_B"
    train_base: bool = False
    mergeable: bool = True
    fan_in_fan_out: bool = False
    weight_layout: WeightLayout | None = None


@dataclass(frozen=True)
class LoRARecipe:
    """Recipe metadata for LoRA fine-tuning."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target: tuple[str, ...] = ("attention.qkv", "attention.proj")
    train_head: bool = True

    @classmethod
    def hard_task(
        cls,
        *,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
        target: tuple[str, ...] = (
            "attention.qkv",
            "attention.proj",
            "mlp.fc1",
            "mlp.fc2",
        ),
        train_head: bool = True,
    ) -> "LoRARecipe":
        """Return the hard-task LoRA recipe preset."""

        return cls(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target=target,
            train_head=train_head,
        )

    @classmethod
    def tiny_data(
        cls,
        *,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        target: tuple[str, ...] = ("attention.qkv", "attention.proj"),
        train_head: bool = True,
    ) -> "LoRARecipe":
        """Return the tiny-data LoRA recipe preset."""

        return cls(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target=target,
            train_head=train_head,
        )

    def to_config(self) -> LoRAConfig:
        """Convert recipe metadata to a LoRA module config."""

        return LoRAConfig(
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target=TargetSpec(tags_any=self.target),
        )


@dataclass(frozen=True)
class RsLoRAConfig(LoRAConfig):
    """Rank-stabilized LoRA configuration."""

    rank: int = 32
    alpha: float = 16.0
    scaling: ScalingMode = "alpha_over_sqrt_r"


@dataclass(frozen=True)
class PiSSAConfig(LoRAConfig):
    """PiSSA initialization configuration for LoRA factors."""

    rank: int = 16
    init: str = "pissa"
    svd: str = "truncated"
    niter: int = 4
    residual_handling: str = "freeze_residual"
    fallback_init: str = "kaiming_A_zero_B"


@dataclass(frozen=True)
class LoRAPlusLabelConfig:
    """Label metadata for LoRA+ A/B learning-rate groups."""

    label_A: str = "lora_A"
    label_B: str = "lora_B"
    label_base: str = "frozen"


@dataclass(frozen=True)
class StaticRankMaskedLoRAConfig(LoRAConfig):
    """Static rank-mask LoRA configuration."""

    rank: int = 12
    initial_rank: int = 12
    target_rank: int = 8
    min_rank: int = 1
    max_rank: int = 16
    rank_mask_init: Literal["all_active", "target_rank"] = "all_active"


@dataclass(frozen=True)
class QuantizedBaseLoRACompatibility:
    """Metadata for applying LoRA around externally quantized linears."""

    allow_lora_on_quantized_linear: bool = True
    quantization_owned_by: Literal["external"] = "external"


@dataclass(frozen=True)
class AdaLoRAMetadata:
    """Static metadata for an AdaLoRA SVD-triplet adapter."""

    logical_id: str = ""
    profile_id: str = "safe_default"


@dataclass(frozen=True)
class AdaLoRAConfig(LoRAConfig):
    """Paper-form AdaLoRA SVD-triplet adapter configuration."""

    rank: int = 12
    alpha: float = 16.0
    scaling: ScalingMode = "alpha_over_r"
    init: str = "adalora_zero_singular"


@dataclass(frozen=True)
class LoRAFAConfig(LoRAConfig):
    """LoRA-FA configuration contract."""

    A_init: Literal["gaussian", "orthonormal_rows"] = "orthonormal_rows"
    gradient_mode: Literal["frozen_A", "corrected_v3"] = "corrected_v3"
    gram_ridge: float = 1e-6
    custom_vjp: bool = True


@dataclass(frozen=True)
class CalibrationSpec:
    """Calibration-data request for data-aware initializers."""

    artifact_kind: str
    sample_count: int | None = None
    data_fingerprint: str | None = None


@dataclass(frozen=True)
class EVAInitializerConfig:
    """Explained Variance Adaptation initializer contract."""

    rank_budget: int
    per_layer_min_rank: int = 0
    per_layer_max_rank: int | None = None
    allocation: Literal["explained_variance"] = "explained_variance"
    activation_centering: bool = False
    svd: Literal["randomized", "full"] = "randomized"
    calibration: CalibrationSpec | None = None


@dataclass(frozen=True)
class QuantizerSpec:
    """External quantizer descriptor for LoftQ initialization."""

    id: str
    bits: int
    format: str
    compute_dtype: str | None = None


@dataclass(frozen=True)
class LoftQConfig:
    """LoftQ initialization contract."""

    rank: int
    quantizer: QuantizerSpec
    iterations: int = 1
    scaling: float = 1.0
    residual_svd: Literal["truncated", "full"] = "truncated"


@dataclass(frozen=True)
class FourierFTConfig:
    """FourierFT sparse spectral delta configuration contract."""

    num_frequencies: int
    scaling: float = 1.0
    seed: int | None = None
    target: TargetSpec = field(default_factory=TargetSpec)


class LoRALinear(eqx.Module):
    """LoRA wrapper for linear-like modules."""

    base: eqx.Module
    lora_A: jax.Array
    lora_B: jax.Array
    rank_mask: jax.Array | None
    base_weight_delta: jax.Array | None
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    scaling_mode: ScalingMode = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    train_base: bool = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)
    fan_in_fan_out: bool = eqx.field(static=True)
    merged: bool = eqx.field(static=True)
    projection_segments: tuple[ProjectionSegment, ...] = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.Module,
        *,
        rank: int,
        alpha: float,
        scaling: ScalingMode,
        dropout: float,
        train_base: bool,
        mergeable: bool,
        key: jax.Array,
        fan_in_fan_out: bool = False,
        lora_A: jax.Array | None = None,
        lora_B: jax.Array | None = None,
        rank_mask: jax.Array | None = None,
        base_weight_delta: jax.Array | None = None,
        merged: bool = False,
        projection_segments: tuple[ProjectionSegment, ...] = (),
    ):
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1.")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling_mode = scaling
        self.dropout = dropout
        self.train_base = train_base
        self.mergeable = mergeable
        self.fan_in_fan_out = fan_in_fan_out
        self.merged = merged
        self.projection_segments = projection_segments

        if lora_A is None or lora_B is None:
            lora_A, lora_B = _init_lora(
                base,
                rank,
                key,
                fan_in_fan_out=fan_in_fan_out,
            )
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.rank_mask = rank_mask
        self.base_weight_delta = base_weight_delta

    @property
    def scaling(self) -> float:
        if self.scaling_mode == "alpha_over_r":
            return float(self.alpha / self.rank)
        if self.scaling_mode == "alpha_over_sqrt_r":
            return float(self.alpha / jnp.sqrt(self.rank))
        raise ValueError(f"Unsupported LoRA scaling mode {self.scaling_mode!r}.")

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        y = self.base(x)
        if self.merged:
            return y
        x_drop = (
            _dropout(x, self.dropout, key)
            if self.dropout > 0.0 and not inference
            else x
        )
        lora_B = self.lora_B
        if self.rank_mask is not None:
            lora_B = lora_B * self.rank_mask[None, :]
        if self.fan_in_fan_out:
            return y + x_drop @ self.delta_weight()
        update = lora_B @ (self.lora_A @ x_drop)
        return y + update * self.scaling

    def delta_weight(self) -> jax.Array:
        """Return the dense LoRA delta in base weight layout."""

        lora_B = self.lora_B
        if self.rank_mask is not None:
            lora_B = lora_B * self.rank_mask[None, :]
        delta = (lora_B @ self.lora_A) * self.scaling
        delta = _mask_projection_segments(delta, self.projection_segments)
        return delta.T if self.fan_in_fan_out else delta

    def merge(self):
        """Return a module with the LoRA delta folded into ``base.weight``."""

        if not self.mergeable:
            raise ValueError("This LoRA module is not mergeable.")
        if self.merged:
            return self
        base = eqx.tree_at(
            lambda m: m.weight,
            self.base,
            self.base.weight + self.delta_weight().astype(self.base.weight.dtype),
        )
        return self._replace(base=base, merged=True)

    def unmerge(self):
        """Return a module with the LoRA delta removed from ``base.weight``."""

        if not self.merged:
            return self
        base = eqx.tree_at(
            lambda m: m.weight,
            self.base,
            self.base.weight - self.delta_weight().astype(self.base.weight.dtype),
        )
        return self._replace(base=base, merged=False)

    def _replace(self, *, base: eqx.Module, merged: bool):
        return self.__class__(
            base,
            rank=self.rank,
            alpha=self.alpha,
            scaling=self.scaling_mode,
            dropout=self.dropout,
            train_base=self.train_base,
            mergeable=self.mergeable,
            fan_in_fan_out=self.fan_in_fan_out,
            key=jr.PRNGKey(0),
            lora_A=self.lora_A,
            lora_B=self.lora_B,
            rank_mask=self.rank_mask,
            base_weight_delta=self.base_weight_delta,
            merged=merged,
            projection_segments=self.projection_segments,
        )


class LoRAMergedLinear(LoRALinear):
    """LoRA wrapper for fused projections such as QKV linears."""


class AdaLoRAModule(eqx.Module):
    """AdaLoRA SVD-triplet wrapper with JIT-stable maximum rank."""

    base: eqx.Module
    P: jax.Array
    singular: jax.Array
    Q: jax.Array
    final_mask: jax.Array | None
    scaling: float = eqx.field(static=True)
    metadata: AdaLoRAMetadata = eqx.field(static=True)
    train_base: bool = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.Module,
        *,
        rank: int,
        alpha: float,
        key: jax.Array,
        train_base: bool = False,
        mergeable: bool = True,
        P: jax.Array | None = None,
        singular: jax.Array | None = None,
        Q: jax.Array | None = None,
        final_mask: jax.Array | None = None,
        metadata: AdaLoRAMetadata | None = None,
    ):
        if rank < 1:
            raise ValueError("AdaLoRA rank must be >= 1.")
        weight = _linear_weight(base)
        out_features, in_features = weight.shape
        key_p, key_q = jr.split(key, 2)
        scale = jnp.asarray(1e-3, dtype=weight.dtype)
        self.base = base
        self.P = (
            jr.normal(key_p, (out_features, rank), dtype=weight.dtype) * scale
            if P is None
            else P
        )
        self.singular = (
            jnp.zeros((rank,), dtype=weight.dtype) if singular is None else singular
        )
        self.Q = (
            jr.normal(key_q, (rank, in_features), dtype=weight.dtype) * scale
            if Q is None
            else Q
        )
        self.final_mask = final_mask
        self.scaling = float(alpha / rank)
        self.metadata = AdaLoRAMetadata() if metadata is None else metadata
        self.train_base = train_base
        self.mergeable = mergeable

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.base(x) + self.delta_weight() @ x

    def delta_weight(self) -> jax.Array:
        singular = self.singular
        if self.final_mask is not None:
            singular = singular * self.final_mask.astype(singular.dtype)
        return (self.P * singular[None, :]) @ self.Q * self.scaling

    def orthogonality_loss(self) -> jax.Array:
        p = self.P.astype(jnp.float32)
        q = self.Q.astype(jnp.float32)
        p_term = p.T @ p - jnp.eye(p.shape[1], dtype=jnp.float32)
        q_term = q @ q.T - jnp.eye(q.shape[0], dtype=jnp.float32)
        return jnp.sum(p_term**2) + jnp.sum(q_term**2)


def apply_lora(
    model: PyTree,
    config: LoRAConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply LoRA wrappers to selected linear modules."""

    config = LoRAConfig() if config is None else config
    module_paths = _target_linear_module_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(module_paths))
    updated = model

    for module_path, subkey in zip(module_paths, keys, strict=True):
        module = get_path(updated, module_path)
        if isinstance(module, LoRALinear):
            continue
        if not _is_linear_like(module):
            raise TypeError(
                f"LoRA target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected a linear-like module."
            )
        wrapper_type = LoRAMergedLinear if _is_fused_qkv_path(module_path) else LoRALinear
        lora_A = lora_B = None
        base_weight_delta = None
        rank = config.rank
        if config.init == "pissa":
            module, lora_A, lora_B, base_weight_delta = _pissa_prepare(
                module,
                config,
            )
            rank = int(lora_A.shape[0])
        elif config.init != "kaiming_A_zero_B":
            raise ValueError(
                f"Unsupported LoRA init {config.init!r}; expected kaiming_A_zero_B or pissa."
            )
        rank_mask = _rank_mask(config, module.weight.dtype)
        lora_module = wrapper_type(
            module,
            rank=rank,
            alpha=config.alpha,
            scaling=config.scaling,
            dropout=config.dropout,
            train_base=config.train_base,
            mergeable=config.mergeable,
            fan_in_fan_out=config.fan_in_fan_out,
            key=subkey,
            lora_A=lora_A,
            lora_B=lora_B,
            rank_mask=rank_mask,
            base_weight_delta=base_weight_delta,
            projection_segments=_projection_segments_for_target(module, config.target),
        )
        updated = eqx.tree_at(
            lambda tree, path=module_path: get_path(tree, path),
            updated,
            lora_module,
        )

    return updated


def apply_adalora(
    model: PyTree,
    config: AdaLoRAConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply paper-form AdaLoRA SVD-triplet wrappers to selected linears."""

    config = AdaLoRAConfig() if config is None else config
    module_paths = _target_linear_module_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(module_paths))
    updated = model
    for module_path, subkey in zip(module_paths, keys, strict=True):
        module = get_path(updated, module_path)
        if isinstance(module, AdaLoRAModule):
            continue
        if not _is_linear_like(module):
            raise TypeError(
                f"AdaLoRA target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected a linear-like module."
            )
        updated = eqx.tree_at(
            lambda tree, path=module_path: get_path(tree, path),
            updated,
            AdaLoRAModule(
                module,
                rank=config.rank,
                alpha=config.alpha,
                train_base=config.train_base,
                mergeable=config.mergeable,
                key=subkey,
                metadata=AdaLoRAMetadata(logical_id=path_to_str(module_path)),
            ),
        )
    return updated


def merge_lora(model: PyTree) -> PyTree:
    """Merge every LoRA module in ``model``."""

    return _map_lora_modules(model, lambda module: module.merge())


def unmerge_lora(model: PyTree) -> PyTree:
    """Unmerge every merged LoRA module in ``model``."""

    return _map_lora_modules(model, lambda module: module.unmerge())


def extract_lora_delta(
    model: PyTree,
    *,
    base_model_name: str | None = None,
    base_checkpoint_id: str | None = None,
    user_metadata: dict[str, Any] | None = None,
) -> FineTuneBundle:
    """Extract a portable LoRA-only delta bundle."""

    entries = []
    for path, module in iter_lora_modules(model):
        entries.append(
            {
                "path": path_to_str(path),
                "class": module.__class__.__name__,
                "rank": module.rank,
                "alpha": module.alpha,
                "scaling": module.scaling_mode,
                "dropout": module.dropout,
                "train_base": module.train_base,
                "mergeable": module.mergeable,
                "fan_in_fan_out": module.fan_in_fan_out,
                "factor_convention": "delta = B @ A",
                "projection_segments": tuple(
                    {
                        "name": segment.name,
                        "axis": segment.axis,
                        "start": segment.start,
                        "stop": segment.stop,
                    }
                    for segment in module.projection_segments
                ),
                "merged": module.merged,
                "base_weight_shape": tuple(_linear_weight(module.base).shape),
                "base_bias_shape": None
                if _linear_bias(module.base) is None
                else tuple(_linear_bias(module.base).shape),
                "lora_A": module.lora_A,
                "lora_B": module.lora_B,
                "rank_mask": module.rank_mask,
                "base_weight_delta": module.base_weight_delta,
            }
        )

    return FineTuneBundle(
        method="lora",
        schema_version=1,
        base_model_name=base_model_name,
        base_checkpoint_id=base_checkpoint_id,
        architecture_hash=architecture_hash(strip_lora(model)),
        adapter_config={"entries": entries},
        trainable_labels=None,
        delta_tree=None,
        metadata={} if user_metadata is None else user_metadata,
    )


def load_lora_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    """Apply a LoRA bundle to a compatible base model."""

    if bundle.method != "lora":
        raise FineTuneBundleError(
            f"Expected a LoRA bundle, got method={bundle.method!r}."
        )

    actual_hash = architecture_hash(base_model)
    if bundle.architecture_hash and bundle.architecture_hash != actual_hash:
        raise FineTuneBundleError(
            "LoRA delta architecture hash mismatch: "
            f"expected {bundle.architecture_hash}, got {actual_hash}."
        )

    updated = base_model
    entries = bundle.adapter_config.get("entries", ())
    for entry in entries:
        path = str_to_path(entry["path"])
        module = _bundle_get_path(updated, path, method_name="LoRA")
        if not _is_linear_like(module):
            raise FineTuneBundleError(
                f"LoRA delta expects linear-like module at {entry['path']}, "
                f"got {type(module).__name__}."
            )
        if tuple(_linear_weight(module).shape) != tuple(entry["base_weight_shape"]):
            raise FineTuneBundleError(
                f"LoRA delta expects path {entry['path']} with weight shape "
                f"{entry['base_weight_shape']}, got {tuple(_linear_weight(module).shape)}."
            )
        expected_bias_shape = entry["base_bias_shape"]
        bias = _linear_bias(module)
        actual_bias_shape = None if bias is None else tuple(bias.shape)
        if actual_bias_shape != expected_bias_shape:
            raise FineTuneBundleError(
                f"LoRA delta expects path {entry['path']} with bias shape "
                f"{expected_bias_shape}, got {actual_bias_shape}."
            )
        base_weight_delta = entry.get("base_weight_delta")
        if base_weight_delta is not None:
            module = eqx.tree_at(
                lambda linear: linear.weight,
                module,
                _linear_weight(module) + base_weight_delta.astype(_linear_weight(module).dtype),
            )

        wrapper_type = LoRAMergedLinear if entry["class"] == "LoRAMergedLinear" else LoRALinear
        lora_module = wrapper_type(
            module,
            rank=int(entry["rank"]),
            alpha=float(entry["alpha"]),
            scaling=entry["scaling"],
            dropout=float(entry["dropout"]),
            train_base=bool(entry["train_base"]),
            mergeable=bool(entry["mergeable"]),
            fan_in_fan_out=bool(entry.get("fan_in_fan_out", False)),
            key=jr.PRNGKey(0),
            lora_A=entry["lora_A"],
            lora_B=entry["lora_B"],
            rank_mask=entry.get("rank_mask"),
            base_weight_delta=base_weight_delta,
            merged=False,
            projection_segments=tuple(
                ProjectionSegment(
                    name=item["name"],
                    axis=int(item["axis"]),
                    start=int(item["start"]),
                    stop=int(item["stop"]),
                )
                for item in entry.get("projection_segments", ())
            ),
        )
        if entry["merged"]:
            lora_module = lora_module.merge()
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, lora_module)

    return updated


def strip_lora(model: PyTree) -> PyTree:
    """Replace LoRA wrappers with their unmerged base linears."""

    stripped = unmerge_lora(model)
    for path, module in iter_lora_modules(stripped):
        base = _restore_base_weight(module)
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, base)
    return stripped


def iter_lora_modules(model: PyTree) -> tuple[tuple[Path, LoRALinear], ...]:
    """Return path/module pairs for LoRA wrappers in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, LoRALinear),
        )
        if isinstance(leaf, LoRALinear)
    )


def lora_rank_groups(model: PyTree) -> dict[str, int]:
    """Return canonical LoRA path strings and their static maximum ranks."""

    return {
        path_to_str(path): module.rank
        for path, module in iter_lora_modules(model)
    }


def apply_lora_rank_pattern(
    model: PyTree,
    rank_pattern: Mapping[str, Any],
    *,
    strict: bool = True,
) -> PyTree:
    """Apply fixed-shape rank masks to LoRA modules by canonical path string."""

    modules = {
        path_to_str(path): (path, module)
        for path, module in iter_lora_modules(model)
    }
    if strict:
        unknown = sorted(set(rank_pattern) - set(modules))
        if unknown:
            raise ValueError(
                "Rank pattern contains unknown LoRA module paths: "
                f"{', '.join(unknown)}."
            )

    updated = model
    for name, value in rank_pattern.items():
        if name not in modules:
            continue
        path, module = modules[name]
        if module.merged:
            raise ValueError(
                f"Cannot apply rank mask for merged LoRA module {name!r}; "
                "unmerge the module before changing rank masks."
            )
        mask = jnp.asarray(value, dtype=jnp.bool_)
        if mask.shape != (module.rank,):
            raise ValueError(
                f"Rank mask for {name!r} must have shape ({module.rank},), "
                f"got {mask.shape}."
            )
        updated = eqx.tree_at(
            lambda tree, p=path: get_path(tree, p),
            updated,
            _replace_lora_rank_mask(module, mask),
        )

    return updated


def architecture_hash(model: PyTree) -> str:
    """Hash parameter paths, shapes, and dtypes for compatibility checks."""

    import hashlib

    digest = hashlib.sha256()
    filtered = eqx.filter(model, eqx.is_inexact_array)
    for key_path, leaf in jtu.tree_leaves_with_path(filtered):
        if not eqx.is_inexact_array(leaf):
            continue
        path = path_to_str(key_path_to_path(key_path))
        digest.update(path.encode())
        digest.update(str(tuple(leaf.shape)).encode())
        digest.update(str(leaf.dtype).encode())
    return digest.hexdigest()


def _target_linear_module_paths(
    model: PyTree,
    target: TargetSpec,
    *,
    tagger: Tagger,
) -> tuple[Path, ...]:
    paths = set()
    resolved = resolve_target(
        model,
        target,
        allow_empty=target.target_kind == "projection_segment",
        tagger=tagger,
    )
    paths.update(_linear_module_path(info.path) for info in resolved)
    if _target_mentions_qkv_segment(target):
        fused = resolve_target(
            model,
            TargetSpec(
                tags_any=("attention.qkv", "block.attention.qkv"),
                allow_empty=True,
            ),
            allow_empty=True,
            tagger=tagger,
        )
        paths.update(_linear_module_path(info.path) for info in fused)
    if not paths and not target.allow_empty:
        raise ValueError("TargetSpec resolved no LoRA linear modules.")
    return tuple(sorted(paths, key=path_to_str))


def _bundle_get_path(model: PyTree, path: Path, *, method_name: str):
    try:
        return get_path(model, path)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise FineTuneBundleError(
            f"{method_name} delta expects path {path_to_str(path)}, "
            "but the base model has no matching leaf."
        ) from error


def _linear_module_path(path: Path) -> Path:
    if path[-1:] in (("weight",), ("bias",)):
        return path[:-1]
    return path


def _is_fused_qkv_path(path: Path) -> bool:
    return "qkv" in {str(part) for part in path}


def _target_mentions_qkv_segment(target: TargetSpec) -> bool:
    tags = set(target.tags_all) | set(target.tags_any)
    suffixes = (".q", ".k", ".v")
    return any(tag in {"attention.q", "attention.k", "attention.v"} or tag.endswith(suffixes) for tag in tags)


def _projection_segments_for_target(
    module: eqx.Module,
    target: TargetSpec,
) -> tuple[ProjectionSegment, ...]:
    if target.target_kind != "projection_segment":
        return ()
    selected = _selected_qkv_segment_names(target)
    if not selected:
        return ()
    weight = _linear_weight(module)
    if weight.shape[0] % 3 != 0:
        raise ValueError("QKV projection segments require an output dimension divisible by 3.")
    width = weight.shape[0] // 3
    starts = {"q": 0, "k": width, "v": 2 * width}
    return tuple(
        ProjectionSegment(name=name, axis=0, start=starts[name], stop=starts[name] + width)
        for name in ("q", "k", "v")
        if name in selected
    )


def _selected_qkv_segment_names(target: TargetSpec) -> frozenset[str]:
    names: set[str] = set()
    for tag in (*target.tags_all, *target.tags_any):
        last = tag.rsplit(".", maxsplit=1)[-1]
        if last in {"q", "k", "v"}:
            names.add(last)
    return frozenset(names)


def _mask_projection_segments(
    delta: jax.Array,
    segments: tuple[ProjectionSegment, ...],
) -> jax.Array:
    if not segments:
        return delta
    mask = jnp.zeros((delta.shape[0],), dtype=delta.dtype)
    for segment in segments:
        if segment.axis != 0:
            raise ValueError("LoRA projection segments currently use logical output axis 0.")
        mask = mask.at[segment.start : segment.stop].set(1)
    return delta * mask[:, None]


def _init_lora(
    base: eqx.Module,
    rank: int,
    key: jax.Array,
    *,
    fan_in_fan_out: bool = False,
) -> tuple[jax.Array, jax.Array]:
    key_a, _ = jr.split(key, 2)
    weight = _linear_weight(base)
    in_features, out_features = (
        (weight.shape[0], weight.shape[1])
        if fan_in_fan_out
        else (weight.shape[1], weight.shape[0])
    )
    bound = jnp.sqrt(6.0 / in_features)
    lora_A = jr.uniform(
        key_a,
        (rank, in_features),
        minval=-bound,
        maxval=bound,
        dtype=weight.dtype,
    )
    lora_B = jnp.zeros((out_features, rank), dtype=weight.dtype)
    return lora_A, lora_B


def _pissa_prepare(
    base: eqx.Module,
    config: LoRAConfig,
) -> tuple[eqx.Module, jax.Array, jax.Array, jax.Array | None]:
    if not isinstance(config, PiSSAConfig):
        lora_A, lora_B = _pissa_init(
            base,
            config.rank,
            scaling=_scaling(config.scaling, config.alpha, config.rank),
            fan_in_fan_out=config.fan_in_fan_out,
        )
        return base, lora_A, lora_B, None
    if config.svd not in {"truncated", "exact"}:
        raise ValueError("PiSSAConfig.svd must be 'truncated' or 'exact'.")
    if config.niter < 0:
        raise ValueError("PiSSAConfig.niter must be non-negative.")
    if config.residual_handling not in {"freeze_residual", "none"}:
        raise ValueError(
            "PiSSAConfig.residual_handling must be 'freeze_residual' or 'none'."
        )
    try:
        lora_A, lora_B = _pissa_init(
            base,
            config.rank,
            scaling=_scaling(config.scaling, config.alpha, config.rank),
            fan_in_fan_out=config.fan_in_fan_out,
        )
    except Exception:
        if config.fallback_init != "kaiming_A_zero_B":
            raise
        lora_A, lora_B = _init_lora(
            base,
            config.rank,
            jr.PRNGKey(0),
            fan_in_fan_out=config.fan_in_fan_out,
        )
        return base, lora_A, lora_B, None
    if config.residual_handling == "none":
        return base, lora_A, lora_B, None

    rank = int(lora_A.shape[0])
    delta = (lora_B @ lora_A) * _scaling(config.scaling, config.alpha, rank)
    if config.fan_in_fan_out:
        delta = delta.T
    weight = _linear_weight(base)
    base_weight_delta = -delta.astype(weight.dtype)
    residual_base = eqx.tree_at(
        lambda linear: linear.weight,
        base,
        weight + base_weight_delta,
    )
    return residual_base, lora_A, lora_B, base_weight_delta


def _pissa_init(
    base: eqx.Module,
    rank: int,
    *,
    scaling: float,
    fan_in_fan_out: bool = False,
) -> tuple[jax.Array, jax.Array]:
    weight = _linear_weight(base)
    svd_weight = weight.T if fan_in_fan_out else weight
    u, s, vh = jnp.linalg.svd(svd_weight, full_matrices=False)
    rank = min(rank, s.shape[0])
    sqrt_s = jnp.sqrt(s[:rank] / scaling).astype(weight.dtype)
    lora_B = u[:, :rank].astype(weight.dtype) * sqrt_s[None, :]
    lora_A = sqrt_s[:, None] * vh[:rank].astype(weight.dtype)
    return lora_A, lora_B


def _scaling(mode: ScalingMode, alpha: float, rank: int) -> float:
    if mode == "alpha_over_r":
        return float(alpha / rank)
    if mode == "alpha_over_sqrt_r":
        return float(alpha / jnp.sqrt(rank))
    raise ValueError(f"Unsupported LoRA scaling mode {mode!r}.")


def _is_linear_like(module: Any) -> bool:
    weight = getattr(module, "weight", None)
    return (
        callable(module)
        and eqx.is_inexact_array(weight)
        and weight.ndim == 2
    )


def _linear_weight(module: Any) -> jax.Array:
    weight = getattr(module, "weight", None)
    if not eqx.is_inexact_array(weight) or weight.ndim != 2:
        raise TypeError(f"{type(module).__name__} is not a linear-like module.")
    return weight


def _linear_bias(module: Any) -> jax.Array | None:
    bias = getattr(module, "bias", None)
    if bias is None:
        return None
    if not eqx.is_inexact_array(bias):
        raise TypeError(f"{type(module).__name__}.bias is not an inexact array.")
    return bias


def _restore_base_weight(module: LoRALinear) -> eqx.Module:
    if module.base_weight_delta is None:
        return module.base
    return eqx.tree_at(
        lambda linear: linear.weight,
        module.base,
        _linear_weight(module.base) - module.base_weight_delta.astype(_linear_weight(module.base).dtype),
    )


def _rank_mask(config: LoRAConfig, dtype) -> jax.Array | None:
    del dtype
    if not isinstance(config, StaticRankMaskedLoRAConfig):
        return None
    if not config.min_rank <= config.target_rank <= config.max_rank:
        raise ValueError("target_rank must lie between min_rank and max_rank.")
    if not config.min_rank <= config.initial_rank <= config.max_rank:
        raise ValueError("initial_rank must lie between min_rank and max_rank.")
    if config.rank_mask_init == "all_active":
        active_rank = min(config.initial_rank, config.rank)
    elif config.rank_mask_init == "target_rank":
        active_rank = min(config.target_rank, config.rank)
    else:
        raise ValueError(
            "rank_mask_init must be either 'all_active' or 'target_rank'."
        )
    values = jnp.arange(config.rank) < active_rank
    return values.astype(jnp.bool_)


def _dropout(x: jax.Array, rate: float, key: jax.Array | None) -> jax.Array:
    if key is None:
        raise ValueError("A PRNG key is required when LoRA dropout is active.")
    keep_prob = 1.0 - rate
    mask = jr.bernoulli(key, keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0)


def _map_lora_modules(model: PyTree, fn) -> PyTree:
    updated = model
    for path, module in iter_lora_modules(updated):
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, fn(module))
    return updated


def _replace_lora_rank_mask(
    module: LoRALinear,
    rank_mask: jax.Array | None,
) -> LoRALinear:
    return module.__class__(
        module.base,
        rank=module.rank,
        alpha=module.alpha,
        scaling=module.scaling_mode,
        dropout=module.dropout,
        train_base=module.train_base,
        mergeable=module.mergeable,
        fan_in_fan_out=module.fan_in_fan_out,
        key=jr.PRNGKey(0),
        lora_A=module.lora_A,
        lora_B=module.lora_B,
        rank_mask=rank_mask,
        base_weight_delta=module.base_weight_delta,
        merged=module.merged,
        projection_segments=module.projection_segments,
    )


def lora_config_to_dict(config: LoRAConfig) -> dict[str, Any]:
    """Serialize a LoRA config without callable selector fields."""

    data = asdict(config)
    target = config.target
    data["target"] = {
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
    return data


__all__ = (
    "LoRAConfig",
    "LoRALinear",
    "LoRAMergedLinear",
    "LoRAPlusLabelConfig",
    "LoRARecipe",
    "PiSSAConfig",
    "QuantizedBaseLoRACompatibility",
    "StaticRankMaskedLoRAConfig",
    "RsLoRAConfig",
    "AdaLoRAConfig",
    "AdaLoRAMetadata",
    "AdaLoRAModule",
    "CalibrationSpec",
    "EVAInitializerConfig",
    "FourierFTConfig",
    "LoftQConfig",
    "LoRAFAConfig",
    "QuantizerSpec",
    "apply_adalora",
    "apply_lora_rank_pattern",
    "apply_lora",
    "architecture_hash",
    "extract_lora_delta",
    "iter_lora_modules",
    "load_lora_delta",
    "lora_config_to_dict",
    "lora_rank_groups",
    "merge_lora",
    "strip_lora",
    "unmerge_lora",
)
