"""VeRA modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..config import TargetSpec
from ..paths import key_path_to_path, path_to_str
from ..selectors import resolve_target
from ..tags import Tagger, canonical_tags_for_path
from .base import get_path


@dataclass(frozen=True)
class VeRAConfig:
    """Configuration for VeRA linear wrappers."""

    rank: int = 256
    shared: bool = True
    seed_required: bool = True
    frozen_A_init: str = "kaiming_uniform"
    frozen_B_init: str = "kaiming_uniform"
    trainable_input_scale_init: float = 1.0
    trainable_output_scale_init: float = 0.0
    mergeable: bool = True
    target: TargetSpec = field(
        default_factory=lambda: TargetSpec(tags_any=("attention.qkv", "attention.proj"))
    )


class VeRALinear(eqx.Module):
    """VeRA wrapper with frozen random bases and trainable scales."""

    base: eqx.nn.Linear
    vera_A: jax.Array
    vera_B: jax.Array
    vera_input_scale: jax.Array
    vera_output_scale: jax.Array
    shared: bool = eqx.field(static=True)
    frozen_A_init: str = eqx.field(static=True)
    frozen_B_init: str = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)
    basis_generation: str = eqx.field(static=True)
    basis_key_data: tuple[int, ...] = eqx.field(static=True)
    basis_pool_key: tuple[str, ...] = eqx.field(static=True)
    share_scope: str = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.nn.Linear,
        *,
        rank: int,
        key: jax.Array,
        shared: bool = True,
        frozen_A_init: str = "kaiming_uniform",
        frozen_B_init: str = "kaiming_uniform",
        input_scale_init: float = 1.0,
        output_scale_init: float = 0.0,
        mergeable: bool = True,
        vera_A: jax.Array | None = None,
        vera_B: jax.Array | None = None,
        vera_input_scale: jax.Array | None = None,
        vera_output_scale: jax.Array | None = None,
        basis_generation: str = "jax.random.PRNGKey_split",
        basis_key_data: tuple[int, ...] = (),
        basis_pool_key: tuple[str, ...] = (),
        share_scope: str = "shape_compatible",
    ):
        key_a, key_b = jr.split(key, 2)
        self.base = base
        self.shared = shared
        self.frozen_A_init = frozen_A_init
        self.frozen_B_init = frozen_B_init
        self.mergeable = mergeable
        self.basis_generation = basis_generation
        self.basis_key_data = basis_key_data
        self.basis_pool_key = basis_pool_key
        self.share_scope = share_scope
        self.vera_A = (
            _init_frozen_matrix(
                key_a,
                (rank, base.in_features),
                frozen_A_init,
                fan_in=base.in_features,
                dtype=base.weight.dtype,
            )
            if vera_A is None
            else vera_A
        )
        self.vera_B = (
            _init_frozen_matrix(
                key_b,
                (base.out_features, rank),
                frozen_B_init,
                fan_in=rank,
                dtype=base.weight.dtype,
            )
            if vera_B is None
            else vera_B
        )
        self.vera_input_scale = (
            jnp.full((rank,), input_scale_init, dtype=base.weight.dtype)
            if vera_input_scale is None
            else vera_input_scale
        )
        self.vera_output_scale = (
            jnp.full((base.out_features,), output_scale_init, dtype=base.weight.dtype)
            if vera_output_scale is None
            else vera_output_scale
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        vera_A = jax.lax.stop_gradient(self.vera_A)
        vera_B = jax.lax.stop_gradient(self.vera_B)
        hidden = (vera_A @ x) * self.vera_input_scale
        update = (vera_B @ hidden) * self.vera_output_scale
        return self.base(x) + update

    def delta_weight(self) -> jax.Array:
        """Return the dense VeRA delta in base weight layout."""

        vera_A = jax.lax.stop_gradient(self.vera_A)
        vera_B = jax.lax.stop_gradient(self.vera_B)
        return (vera_B * self.vera_output_scale[:, None]) @ (
            vera_A * self.vera_input_scale[:, None]
        )


def apply_vera(
    model: PyTree,
    config: VeRAConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply VeRA wrappers to selected linear modules."""

    config = VeRAConfig() if config is None else config
    module_paths = _target_linear_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(module_paths))
    shared_bases: dict[
        tuple[int, int, int, str, str, str],
        tuple[jax.Array, jax.Array, tuple[int, ...]],
    ] = {}
    updated = model
    for module_path, subkey in zip(module_paths, keys, strict=True):
        module = get_path(updated, module_path)
        if isinstance(module, VeRALinear):
            continue
        if not isinstance(module, eqx.nn.Linear):
            raise TypeError(
                f"VeRA target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected eqx.nn.Linear."
            )
        vera_A = vera_B = None
        basis_key_data = _key_data_tuple(subkey)
        basis_pool_key = _vera_pool_key(module, config)
        share_scope = "shape_compatible" if config.shared else "per_module"
        if config.shared:
            vera_A, vera_B, basis_key_data = _shared_vera_bases(
                module,
                config,
                subkey,
                shared_bases,
            )
        updated = eqx.tree_at(
            lambda tree, p=module_path: get_path(tree, p),
            updated,
            VeRALinear(
                module,
                rank=config.rank,
                key=subkey,
                shared=config.shared,
                frozen_A_init=config.frozen_A_init,
                frozen_B_init=config.frozen_B_init,
                input_scale_init=config.trainable_input_scale_init,
                output_scale_init=config.trainable_output_scale_init,
                mergeable=config.mergeable,
                vera_A=vera_A,
                vera_B=vera_B,
                basis_generation="jax.random.PRNGKey_split",
                basis_key_data=basis_key_data,
                basis_pool_key=basis_pool_key,
                share_scope=share_scope,
            ),
        )
    return updated


def merge_vera(model: PyTree) -> PyTree:
    """Fold every mergeable VeRA wrapper into its base linear module."""

    updated = model
    for path, module in iter_vera_modules(updated):
        if not module.mergeable:
            raise ValueError("This VeRA module is not mergeable.")
        base = eqx.tree_at(
            lambda linear: linear.weight,
            module.base,
            module.base.weight + module.delta_weight().astype(module.base.weight.dtype),
        )
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, base)
    return updated


def iter_vera_modules(model: PyTree) -> tuple[tuple[Path, VeRALinear], ...]:
    """Return path/module pairs for VeRA wrappers in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, VeRALinear),
        )
        if isinstance(leaf, VeRALinear)
    )


def strip_vera(model: PyTree) -> PyTree:
    """Replace VeRA wrappers with their base linears."""

    stripped = model
    for path, module in iter_vera_modules(stripped):
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, module.base)
    return stripped


def _target_linear_paths(model: PyTree, target: TargetSpec, *, tagger: Tagger) -> tuple[Path, ...]:
    paths = {
        info.path[:-1]
        for info in resolve_target(model, target, tagger=tagger)
        if info.path[-1:] in (("weight",), ("bias",))
    }
    return tuple(sorted(paths, key=path_to_str))


def _shared_vera_bases(
    module: eqx.nn.Linear,
    config: VeRAConfig,
    key: jax.Array,
    cache: dict[
        tuple[int, int, int, str, str, str],
        tuple[jax.Array, jax.Array, tuple[int, ...]],
    ],
) -> tuple[jax.Array, jax.Array, tuple[int, ...]]:
    cache_key = _vera_pool_key(module, config)
    if cache_key not in cache:
        key_a, key_b = jr.split(key, 2)
        cache[cache_key] = (
            _init_frozen_matrix(
                key_a,
                (config.rank, module.in_features),
                config.frozen_A_init,
                fan_in=module.in_features,
                dtype=module.weight.dtype,
            ),
            _init_frozen_matrix(
                key_b,
                (module.out_features, config.rank),
                config.frozen_B_init,
                fan_in=config.rank,
                dtype=module.weight.dtype,
            ),
            _key_data_tuple(key),
        )
    return cache[cache_key]


def _vera_pool_key(
    module: eqx.nn.Linear,
    config: VeRAConfig,
) -> tuple[int, int, int, str, str, str]:
    return (
        int(config.rank),
        int(module.in_features),
        int(module.out_features),
        str(module.weight.dtype),
        config.frozen_A_init,
        config.frozen_B_init,
    )


def _key_data_tuple(key: jax.Array) -> tuple[int, ...]:
    try:
        data = jr.key_data(key)
    except AttributeError:
        data = key
    return tuple(int(part) for part in jnp.ravel(jnp.asarray(data)))


def _init_frozen_matrix(
    key: jax.Array,
    shape: tuple[int, int],
    init: str,
    *,
    fan_in: int,
    dtype,
) -> jax.Array:
    if init == "kaiming_uniform":
        bound = jnp.sqrt(6.0 / fan_in)
        return jr.uniform(key, shape, minval=-bound, maxval=bound, dtype=dtype)
    if init == "normal":
        return jr.normal(key, shape, dtype=dtype) / jnp.sqrt(fan_in)
    raise ValueError(
        f"Unsupported VeRA frozen matrix init {init!r}; expected kaiming_uniform or normal."
    )


__all__ = (
    "VeRAConfig",
    "VeRALinear",
    "apply_vera",
    "iter_vera_modules",
    "merge_vera",
    "strip_vera",
)
