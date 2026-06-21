"""LoRA modules and model surgery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..config import FineTuneBundle, TargetSpec
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
            tags=("attention.qkv", "attention.proj"),
        )
    )
    init: str = "kaiming_A_zero_B"
    train_base: bool = False
    mergeable: bool = True


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


@dataclass(frozen=True)
class RankMaskedLoRAConfig(LoRAConfig):
    """Static rank-mask LoRA configuration."""

    rank: int = 12
    initial_rank: int = 12
    target_rank: int = 8
    min_rank: int = 1
    max_rank: int = 16


class LoRALinear(eqx.Module):
    """LoRA wrapper for ``eqx.nn.Linear``."""

    base: eqx.nn.Linear
    lora_A: jax.Array
    lora_B: jax.Array
    rank_mask: jax.Array | None
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    scaling_mode: ScalingMode = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    train_base: bool = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)
    merged: bool = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.nn.Linear,
        *,
        rank: int,
        alpha: float,
        scaling: ScalingMode,
        dropout: float,
        train_base: bool,
        mergeable: bool,
        key: jax.Array,
        lora_A: jax.Array | None = None,
        lora_B: jax.Array | None = None,
        rank_mask: jax.Array | None = None,
        merged: bool = False,
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
        self.merged = merged

        if lora_A is None or lora_B is None:
            lora_A, lora_B = _init_lora(base, rank, key)
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.rank_mask = rank_mask

    @property
    def scaling(self) -> float:
        if self.scaling_mode == "alpha_over_r":
            return float(self.alpha / self.rank)
        if self.scaling_mode == "alpha_over_sqrt_r":
            return float(self.alpha / jnp.sqrt(self.rank))
        raise ValueError(f"Unsupported LoRA scaling mode {self.scaling_mode!r}.")

    def __call__(self, x: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        y = self.base(x)
        if self.merged:
            return y
        x_drop = _dropout(x, self.dropout, key) if self.dropout > 0.0 else x
        update = self.lora_B @ (self.lora_A @ x_drop)
        return y + update * self.scaling

    def delta_weight(self) -> jax.Array:
        """Return the dense LoRA delta in base weight layout."""

        lora_B = self.lora_B
        if self.rank_mask is not None:
            lora_B = lora_B * self.rank_mask[None, :]
        return (lora_B @ self.lora_A) * self.scaling

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

    def _replace(self, *, base: eqx.nn.Linear, merged: bool):
        return self.__class__(
            base,
            rank=self.rank,
            alpha=self.alpha,
            scaling=self.scaling_mode,
            dropout=self.dropout,
            train_base=self.train_base,
            mergeable=self.mergeable,
            key=jr.PRNGKey(0),
            lora_A=self.lora_A,
            lora_B=self.lora_B,
            rank_mask=self.rank_mask,
            merged=merged,
        )


class LoRAMergedLinear(LoRALinear):
    """LoRA wrapper for fused projections such as QKV linears."""


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
        if not isinstance(module, eqx.nn.Linear):
            raise TypeError(
                f"LoRA target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected eqx.nn.Linear."
            )
        wrapper_type = LoRAMergedLinear if _is_fused_qkv_path(module_path) else LoRALinear
        lora_A = lora_B = None
        if config.init == "pissa":
            lora_A, lora_B = _pissa_init(module, config.rank)
        elif config.init != "kaiming_A_zero_B":
            raise ValueError(
                f"Unsupported LoRA init {config.init!r}; expected kaiming_A_zero_B or pissa."
            )
        rank_mask = _rank_mask(config, module.weight.dtype)
        lora_module = wrapper_type(
            module,
            rank=config.rank,
            alpha=config.alpha,
            scaling=config.scaling,
            dropout=config.dropout,
            train_base=config.train_base,
            mergeable=config.mergeable,
            key=subkey,
            lora_A=lora_A,
            lora_B=lora_B,
            rank_mask=rank_mask,
        )
        updated = eqx.tree_at(
            lambda tree, path=module_path: get_path(tree, path),
            updated,
            lora_module,
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
                "merged": module.merged,
                "base_weight_shape": tuple(module.base.weight.shape),
                "base_bias_shape": None
                if module.base.bias is None
                else tuple(module.base.bias.shape),
                "lora_A": module.lora_A,
                "lora_B": module.lora_B,
                "rank_mask": module.rank_mask,
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
        raise ValueError(f"Expected a LoRA bundle, got method={bundle.method!r}.")

    actual_hash = architecture_hash(base_model)
    if bundle.architecture_hash and bundle.architecture_hash != actual_hash:
        raise ValueError(
            "LoRA delta architecture hash mismatch: "
            f"expected {bundle.architecture_hash}, got {actual_hash}."
        )

    updated = base_model
    entries = bundle.adapter_config.get("entries", ())
    for entry in entries:
        path = str_to_path(entry["path"])
        module = get_path(updated, path)
        if not isinstance(module, eqx.nn.Linear):
            raise ValueError(
                f"LoRA delta expects linear module at {entry['path']}, "
                f"got {type(module).__name__}."
            )
        if tuple(module.weight.shape) != tuple(entry["base_weight_shape"]):
            raise ValueError(
                f"LoRA delta expects path {entry['path']} with weight shape "
                f"{entry['base_weight_shape']}, got {tuple(module.weight.shape)}."
            )
        expected_bias_shape = entry["base_bias_shape"]
        actual_bias_shape = None if module.bias is None else tuple(module.bias.shape)
        if actual_bias_shape != expected_bias_shape:
            raise ValueError(
                f"LoRA delta expects path {entry['path']} with bias shape "
                f"{expected_bias_shape}, got {actual_bias_shape}."
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
            key=jr.PRNGKey(0),
            lora_A=entry["lora_A"],
            lora_B=entry["lora_B"],
            rank_mask=entry.get("rank_mask"),
            merged=False,
        )
        if entry["merged"]:
            lora_module = lora_module.merge()
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, lora_module)

    return updated


def strip_lora(model: PyTree) -> PyTree:
    """Replace LoRA wrappers with their unmerged base linears."""

    stripped = unmerge_lora(model)
    for path, module in iter_lora_modules(stripped):
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, module.base)
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
    paths = {
        _linear_module_path(info.path)
        for info in resolve_target(model, target, allow_empty=False, tagger=tagger)
    }
    return tuple(sorted(paths, key=path_to_str))


def _linear_module_path(path: Path) -> Path:
    if path[-1:] in (("weight",), ("bias",)):
        return path[:-1]
    return path


def _is_fused_qkv_path(path: Path) -> bool:
    return "qkv" in {str(part) for part in path}


def _init_lora(
    base: eqx.nn.Linear,
    rank: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    key_a, _ = jr.split(key, 2)
    in_features = base.weight.shape[1]
    out_features = base.weight.shape[0]
    bound = jnp.sqrt(6.0 / in_features)
    lora_A = jr.uniform(
        key_a,
        (rank, in_features),
        minval=-bound,
        maxval=bound,
        dtype=base.weight.dtype,
    )
    lora_B = jnp.zeros((out_features, rank), dtype=base.weight.dtype)
    return lora_A, lora_B


def _pissa_init(
    base: eqx.nn.Linear,
    rank: int,
) -> tuple[jax.Array, jax.Array]:
    u, s, vh = jnp.linalg.svd(base.weight, full_matrices=False)
    rank = min(rank, s.shape[0])
    sqrt_s = jnp.sqrt(s[:rank]).astype(base.weight.dtype)
    lora_B = u[:, :rank].astype(base.weight.dtype) * sqrt_s[None, :]
    lora_A = sqrt_s[:, None] * vh[:rank].astype(base.weight.dtype)
    return lora_A, lora_B


def _rank_mask(config: LoRAConfig, dtype) -> jax.Array | None:
    if not isinstance(config, RankMaskedLoRAConfig):
        return None
    if not config.min_rank <= config.target_rank <= config.max_rank:
        raise ValueError("target_rank must lie between min_rank and max_rank.")
    if config.target_rank > config.rank:
        raise ValueError("target_rank must be <= rank.")
    values = jnp.arange(config.rank) < config.target_rank
    return values.astype(dtype)


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


def lora_config_to_dict(config: LoRAConfig) -> dict[str, Any]:
    """Serialize a LoRA config without callable selector fields."""

    data = asdict(config)
    target = config.target
    data["target"] = {
        "include": target.include,
        "exclude": target.exclude,
        "tags": target.tags,
        "min_depth": target.min_depth,
        "max_depth": target.max_depth,
        "predicate": None
        if target.predicate is None
        else getattr(target.predicate, "__name__", "<callable>"),
    }
    return data


__all__ = (
    "LoRAConfig",
    "LoRALinear",
    "LoRAMergedLinear",
    "PiSSAConfig",
    "RankMaskedLoRAConfig",
    "RsLoRAConfig",
    "apply_lora",
    "architecture_hash",
    "extract_lora_delta",
    "iter_lora_modules",
    "load_lora_delta",
    "lora_config_to_dict",
    "merge_lora",
    "strip_lora",
    "unmerge_lora",
)
