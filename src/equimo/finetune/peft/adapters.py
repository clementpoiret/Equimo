"""Bottleneck adapter modules and model surgery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..config import FineTuneBundle, FineTuneBundleError, TargetSpec, TrainableSpec
from ..heads import ActivationName
from ..paths import key_path_to_path, path_to_str, str_to_path
from ..selectors import resolve_target
from ..tags import Tagger, canonical_tags_for_path
from .base import get_path
from .lora import architecture_hash


AdapterPlacement = Literal["after_mlp", "parallel", "both"]
AdapterMissingPolicy = Literal["error", "ignore"]
ActiveAdapter = str | tuple[str, ...]


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for bottleneck adapters."""

    bottleneck: int | None = None
    reduction_factor: int = 16
    bottleneck_min: int = 16
    bottleneck_max: int = 64
    placement: AdapterPlacement = "after_mlp"
    activation: ActivationName = "gelu"
    dropout: float = 0.0
    down_init: str = "kaiming_uniform"
    up_init: str = "zeros"
    residual_scale_init: float = 1.0
    pre_norm: bool = False
    train_base: bool = False
    target: TargetSpec = TargetSpec(tags_any=("block",))


@dataclass(frozen=True)
class AdapterRecipe:
    """Recipe metadata for bottleneck adapter fine-tuning."""

    bottleneck: int = 64
    placement: AdapterPlacement = "after_mlp"
    activation: ActivationName = "gelu"
    dropout: float = 0.0
    train_head: bool = True
    train_norm: bool = True

    @classmethod
    def strong(
        cls,
        *,
        bottleneck: int = 64,
        placement: AdapterPlacement = "both",
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
        train_head: bool = True,
        train_norm: bool = True,
    ) -> "AdapterRecipe":
        """Return the stronger two-placement adapter recipe preset."""

        return cls(
            bottleneck=bottleneck,
            placement=placement,
            activation=activation,
            dropout=dropout,
            train_head=train_head,
            train_norm=train_norm,
        )

    def to_config(self) -> AdapterConfig:
        """Convert recipe metadata to an adapter module config."""

        return AdapterConfig(
            bottleneck=self.bottleneck,
            placement=self.placement,
            activation=self.activation,
            dropout=self.dropout,
        )


@dataclass(frozen=True)
class ParallelAdapterConfig(AdapterConfig):
    """Configuration for residual parallel adapters."""

    bottleneck: int | None = 64
    placement: Literal["parallel"] = "parallel"
    branch: str = "mlp"
    fusion: Literal["residual_sum"] = "residual_sum"


@dataclass(frozen=True)
class AdaptFormerConfig:
    """Configuration for AdaptFormer-style parallel adapters."""

    bottleneck: int = 64
    placement: Literal["parallel_mlp"] = "parallel_mlp"
    activation: ActivationName = "gelu"
    dropout: float = 0.0
    up_init: str = "zeros"
    scale_init: float = 1.0
    scale_trainable: bool = True
    train_head: bool = True
    target: TargetSpec = TargetSpec(tags_any=("block",))

    @classmethod
    def safe_default(
        cls,
        *,
        bottleneck: int = 64,
        target: TargetSpec = TargetSpec(tags_any=("block",)),
    ) -> "AdaptFormerConfig":
        """Return Equimo's identity-preserving AdaptFormer recipe."""

        return cls(bottleneck=bottleneck, target=target)

    @classmethod
    def paper_chen2022(
        cls,
        *,
        bottleneck: int = 64,
        target: TargetSpec = TargetSpec(tags_any=("block",)),
    ) -> "AdaptFormerConfig":
        """Return the Chen et al. AdaptFormer paper profile."""

        return cls(
            bottleneck=bottleneck,
            activation="relu",
            scale_init=0.1,
            scale_trainable=False,
            target=target,
        )


@dataclass(frozen=True)
class AdapterBankConfig:
    """Configuration for named adapter-bank selection."""

    active: ActiveAdapter = "default"
    allow_multiple_active: bool = False
    missing_adapter_policy: AdapterMissingPolicy = "error"


@dataclass(frozen=True)
class AdapterFusionConfig:
    """Metadata for AdapterFusion-style adapter composition."""

    fusion: Literal["attention"] = "attention"
    freeze_task_adapters: bool = True
    fusion_dropout: float = 0.0
    placement: tuple[str, ...] = ("after_mlp",)


@dataclass(frozen=True)
class OrthogonalAdapterConfig:
    """OFT/BOFT orthogonal adaptation configuration contract."""

    side: Literal["input", "output"] = "input"
    parameterization: Literal["cayley", "butterfly_cayley"] = "cayley"
    block_size: int | None = None
    num_factors: int = 1
    eps: float = 1e-6
    train_base: bool = False
    mergeable: bool = True
    target: TargetSpec = TargetSpec(tags_any=("attention.qkv", "attention.proj", "mlp.fc1", "mlp.fc2"))

    def __post_init__(self) -> None:
        if self.num_factors < 1:
            raise ValueError("OrthogonalAdapterConfig.num_factors must be positive.")


@dataclass(frozen=True)
class ConvPassConfig:
    """Vision-native ConvPass adapter configuration."""

    bottleneck: int = 8
    placement: tuple[Literal["attention", "mlp"], ...] = ("attention", "mlp")
    spatial_kernel: int = 3
    activation: ActivationName = "gelu"
    dropout: float = 0.0
    up_init: str = "zeros"
    class_token_policy: Literal["separate", "bypass", "exclude"] = "separate"
    patch_grid: tuple[int, int] | None = None
    train_base: bool = False
    target: TargetSpec = TargetSpec(tags_any=("block",))


@dataclass(frozen=True)
class RepAdapterConfig:
    """Structurally reparameterizable linear adapter configuration."""

    bottleneck: int = 8
    residual_scale_init: float = 1.0
    down_init: str = "kaiming_uniform"
    up_init: str = "zeros"
    branch_bias: bool = False
    train_base: bool = False
    mergeable: bool = True
    target: TargetSpec = TargetSpec(
        tags_any=("attention.proj", "mlp.fc1", "mlp.fc2")
    )


class AdapterFusion(eqx.Module):
    """Attention fusion over active adapter outputs."""

    scorer: eqx.nn.Linear
    fusion: Literal["attention"] = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_adapters: int,
        *,
        key: jax.Array,
        fusion: Literal["attention"] = "attention",
        dropout: float = 0.0,
        scorer: eqx.nn.Linear | None = None,
    ):
        if fusion != "attention":
            raise ValueError("AdapterFusion currently supports fusion='attention'.")
        if num_adapters < 1:
            raise ValueError("AdapterFusion requires at least one adapter.")
        self.scorer = (
            _zero_linear(eqx.nn.Linear(dim, num_adapters, key=key))
            if scorer is None
            else scorer
        )
        self.fusion = fusion
        self.dropout = dropout

    def attention_weights(self, x: jax.Array, count: int) -> jax.Array:
        """Return attention weights for ``count`` adapter outputs."""

        logits = _apply_last_axis(self.scorer, x)[..., :count]
        return jax.nn.softmax(logits, axis=-1)

    def __call__(
        self,
        x: jax.Array,
        adapter_outputs: tuple[jax.Array, ...],
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        if not adapter_outputs:
            raise ValueError("AdapterFusion requires at least one adapter output.")
        weights = self.attention_weights(x, len(adapter_outputs))
        if self.dropout > 0.0 and not inference:
            if key is None:
                raise ValueError("A PRNG key is required when adapter fusion dropout is active.")
            weights = _dropout(weights, self.dropout, key)
        stacked = jnp.stack(adapter_outputs, axis=-2)
        return jnp.sum(stacked * weights[..., None], axis=-2)


class BottleneckAdapter(eqx.Module):
    """Residual bottleneck adapter with identity-preserving zero-up init."""

    norm: eqx.nn.LayerNorm | None
    down: eqx.nn.Linear
    up: eqx.nn.Linear
    residual_scale: jax.Array
    activation: ActivationName = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        bottleneck: int,
        *,
        key: jax.Array,
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
        down_init: str = "kaiming_uniform",
        up_init: str = "zeros",
        residual_scale_init: float = 1.0,
        pre_norm: bool = False,
        norm: eqx.nn.LayerNorm | None = None,
        down: eqx.nn.Linear | None = None,
        up: eqx.nn.Linear | None = None,
        residual_scale: jax.Array | None = None,
    ):
        key_down, key_up = jr.split(key, 2)
        self.norm = (
            eqx.nn.LayerNorm(dim)
            if norm is None and pre_norm
            else norm
        )
        self.down = (
            _init_linear(
                eqx.nn.Linear(dim, bottleneck, key=key_down),
                key_down,
                down_init,
            )
            if down is None
            else down
        )
        self.up = (
            _init_linear(
                eqx.nn.Linear(bottleneck, dim, key=key_up),
                key_up,
                up_init,
            )
            if up is None
            else up
        )
        self.residual_scale = (
            jnp.asarray(residual_scale_init, dtype=jnp.float32)
            if residual_scale is None
            else residual_scale
        )
        self.activation = activation
        self.dropout = dropout

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        y = _apply_last_axis(self.norm, x) if self.norm is not None else x
        y = _apply_last_axis(self.down, y)
        y = _activation(self.activation)(y)
        if self.dropout > 0.0 and not inference:
            if key is None:
                raise ValueError("A PRNG key is required when adapter dropout is active.")
            y = _dropout(y, self.dropout, key)
        return _apply_last_axis(self.up, y) * self.residual_scale


class AdaptFormerAdapter(eqx.Module):
    """AdaptFormer adapter branch."""

    adapter: BottleneckAdapter
    scale: jax.Array

    def __init__(
        self,
        dim: int,
        bottleneck: int,
        *,
        key: jax.Array,
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
        up_init: str = "zeros",
        scale_init: float = 1.0,
        adapter: BottleneckAdapter | None = None,
        scale: jax.Array | None = None,
    ):
        self.adapter = (
            BottleneckAdapter(
                dim,
                bottleneck,
                key=key,
                activation=activation,
                dropout=dropout,
                up_init=up_init,
            )
            if adapter is None
            else adapter
        )
        self.scale = jnp.asarray(scale_init, dtype=jnp.float32) if scale is None else scale

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        return self.adapter(x, key=key, inference=inference) * self.scale


class SerialAdapterBlock(eqx.Module):
    """Wrap a block and add residual adapters after the block output."""

    base: eqx.Module
    adapters: tuple[BottleneckAdapter, ...]
    adapter_names: tuple[str, ...] = eqx.field(static=True, default=())
    active_adapter: ActiveAdapter | None = eqx.field(static=True, default=None)
    adapter_fusion: AdapterFusion | None = None
    train_base: bool = eqx.field(static=True, default=False)

    def __call__(self, x: jax.Array, *args, key: jax.Array | None = None, inference: bool | None = None, **kwargs):
        y = _call_base(self.base, x, *args, key=key, inference=inference, **kwargs)
        adapters = _active_adapters(self)
        if self.adapter_fusion is not None:
            keys = _split_optional_key(key, len(adapters) + 1)
            adapter_outputs = tuple(
                adapter(y, key=adapter_key, inference=inference)
                for adapter, adapter_key in zip(adapters, keys[:-1], strict=True)
            )
            return y + self.adapter_fusion(
                y,
                adapter_outputs,
                key=keys[-1],
                inference=inference,
            )
        keys = _split_optional_key(key, len(adapters))
        for adapter, adapter_key in zip(adapters, keys, strict=True):
            y = y + adapter(y, key=adapter_key, inference=inference)
        return y


class OutputAdapterModule(eqx.Module):
    """Wrap a module and add residual adapters after its output."""

    base: eqx.Module
    adapters: tuple[BottleneckAdapter, ...]
    adapter_names: tuple[str, ...] = eqx.field(static=True, default=())
    active_adapter: ActiveAdapter | None = eqx.field(static=True, default=None)
    adapter_fusion: AdapterFusion | None = None
    train_base: bool = eqx.field(static=True, default=False)

    def __call__(self, x: jax.Array, *args, key: jax.Array | None = None, inference: bool | None = None, **kwargs):
        y = _call_base(self.base, x, *args, key=key, inference=inference, **kwargs)
        adapters = _active_adapters(self)
        if self.adapter_fusion is not None:
            keys = _split_optional_key(key, len(adapters) + 1)
            adapter_outputs = tuple(
                adapter(y, key=adapter_key, inference=inference)
                for adapter, adapter_key in zip(adapters, keys[:-1], strict=True)
            )
            return y + self.adapter_fusion(
                y,
                adapter_outputs,
                key=keys[-1],
                inference=inference,
            )
        keys = _split_optional_key(key, len(adapters))
        for adapter, adapter_key in zip(adapters, keys, strict=True):
            y = y + adapter(y, key=adapter_key, inference=inference)
        return y


class ParallelAdapterBlock(eqx.Module):
    """Wrap a block and add an adapter branch from the block input."""

    base: eqx.Module
    adapter: BottleneckAdapter
    train_base: bool = eqx.field(static=True, default=False)

    def __call__(self, x: jax.Array, *args, key: jax.Array | None = None, inference: bool | None = None, **kwargs):
        key_base, key_adapter = _split_pair(key)
        y = _call_base(self.base, x, *args, key=key_base, inference=inference, **kwargs)
        return y + self.adapter(x, key=key_adapter, inference=inference)


class AdaptFormerBlock(eqx.Module):
    """Wrap a block with an AdaptFormer-style parallel branch."""

    base: eqx.Module
    adapter: AdaptFormerAdapter

    def __call__(self, x: jax.Array, *args, key: jax.Array | None = None, inference: bool | None = None, **kwargs):
        key_base, key_adapter = _split_pair(key)
        y = _call_base(self.base, x, *args, key=key_base, inference=inference, **kwargs)
        return y + self.adapter(x, key=key_adapter, inference=inference)


class OrthogonalLinear(eqx.Module):
    """OFT/BOFT-style orthogonal wrapper for a linear-like module."""

    base: eqx.Module
    skew: jax.Array
    side: Literal["input", "output"] = eqx.field(static=True)
    parameterization: Literal["cayley", "butterfly_cayley"] = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    train_base: bool = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)
    merged: bool = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.Module,
        *,
        side: Literal["input", "output"],
        parameterization: Literal["cayley", "butterfly_cayley"],
        block_size: int | None,
        num_factors: int,
        eps: float,
        train_base: bool,
        mergeable: bool,
        skew: jax.Array | None = None,
        merged: bool = False,
    ):
        weight = _linear_weight(base)
        dim = weight.shape[1] if side == "input" else weight.shape[0]
        if parameterization == "butterfly_cayley" and block_size is None:
            raise ValueError("BOFT butterfly_cayley requires an explicit block_size.")
        if block_size is not None and dim % block_size != 0:
            raise ValueError("OrthogonalAdapterConfig.block_size must divide the adapted dimension.")
        self.base = base
        self.skew = (
            jnp.zeros(
                _orthogonal_skew_shape(
                    dim,
                    block_size,
                    num_factors=(
                        num_factors
                        if parameterization == "butterfly_cayley"
                        else 1
                    ),
                ),
                dtype=weight.dtype,
            )
            if skew is None
            else skew
        )
        self.side = side
        self.parameterization = parameterization
        self.block_size = block_size
        self.num_factors = num_factors
        self.eps = eps
        self.train_base = train_base
        self.mergeable = mergeable
        self.merged = merged

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.merged:
            return self.base(x)
        if self.parameterization == "butterfly_cayley":
            if self.side == "input":
                return self.base(_apply_boft_to_last_axis(x, self.skew, eps=self.eps))
            y = _linear_call(_linear_weight(self.base), None, x)
            y = _apply_boft_to_last_axis(y, self.skew, eps=self.eps)
            bias = _linear_bias(self.base)
            return y if bias is None else y + bias
        return _linear_call(self.effective_weight(), _linear_bias(self.base), x)

    def orthogonal_matrix(self) -> jax.Array:
        """Return the current orthogonal transform."""

        return _orthogonal_from_skew(self.skew, eps=self.eps, block_size=self.block_size)

    def effective_weight(self) -> jax.Array:
        """Return the dense effective weight."""

        weight = _linear_weight(self.base)
        R = self.orthogonal_matrix().astype(weight.dtype)
        if self.side == "input":
            return weight @ R
        return R @ weight

    def delta_weight(self) -> jax.Array:
        """Return the dense delta implied by the orthogonal transform."""

        return self.effective_weight() - _linear_weight(self.base)

    def orthogonality_error(self) -> jax.Array:
        """Return ``||R^T R - I||_F^2`` in fp32."""

        R = self.orthogonal_matrix().astype(jnp.float32)
        eye = jnp.eye(R.shape[0], dtype=jnp.float32)
        return jnp.sum((R.T @ R - eye) ** 2)

    def merge(self):
        """Return a module with the orthogonal transform folded into weight."""

        if not self.mergeable:
            raise ValueError("This orthogonal adapter is not mergeable.")
        if self.merged:
            raise ValueError("Orthogonal adapter is already merged.")
        base = eqx.tree_at(
            lambda linear: linear.weight,
            self.base,
            self.effective_weight().astype(_linear_weight(self.base).dtype),
        )
        return self._replace(base=base, merged=True)

    def unmerge(self):
        """Return a module with the orthogonal transform removed from weight."""

        if not self.merged:
            return self
        R = self.orthogonal_matrix().astype(_linear_weight(self.base).dtype)
        unmerged_weight = (
            _linear_weight(self.base) @ R.T
            if self.side == "input"
            else R.T @ _linear_weight(self.base)
        )
        base = eqx.tree_at(
            lambda linear: linear.weight,
            self.base,
            unmerged_weight,
        )
        return self._replace(base=base, merged=False)

    def _replace(self, *, base: eqx.Module, merged: bool):
        return OrthogonalLinear(
            base,
            side=self.side,
            parameterization=self.parameterization,
            block_size=self.block_size,
            num_factors=self.num_factors,
            eps=self.eps,
            train_base=self.train_base,
            mergeable=self.mergeable,
            skew=self.skew,
            merged=merged,
        )


class RepAdapterLinear(eqx.Module):
    """Linear RepAdapter branch that can be exactly folded into the host weight."""

    base: eqx.Module
    down: eqx.nn.Linear
    up: eqx.nn.Linear
    residual_scale: jax.Array
    train_base: bool = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)
    merged: bool = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.Module,
        *,
        bottleneck: int,
        residual_scale_init: float,
        down_init: str,
        up_init: str,
        branch_bias: bool,
        train_base: bool,
        mergeable: bool,
        key: jax.Array,
        down: eqx.nn.Linear | None = None,
        up: eqx.nn.Linear | None = None,
        residual_scale: jax.Array | None = None,
        merged: bool = False,
    ):
        if bottleneck < 1:
            raise ValueError("RepAdapter bottleneck must be >= 1.")
        weight = _linear_weight(base)
        out_features, in_features = weight.shape
        key_down, key_up = jr.split(key, 2)
        self.base = base
        self.down = (
            _init_linear(
                eqx.nn.Linear(
                    in_features,
                    bottleneck,
                    use_bias=branch_bias,
                    key=key_down,
                ),
                key_down,
                down_init,
            )
            if down is None
            else down
        )
        self.up = (
            _init_linear(
                eqx.nn.Linear(
                    bottleneck,
                    out_features,
                    use_bias=branch_bias,
                    key=key_up,
                ),
                key_up,
                up_init,
            )
            if up is None
            else up
        )
        self.residual_scale = (
            jnp.asarray(residual_scale_init, dtype=weight.dtype)
            if residual_scale is None
            else residual_scale
        )
        self.train_base = train_base
        self.mergeable = mergeable
        self.merged = merged

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x)
        if self.merged:
            return y
        return y + _apply_last_axis(self.up, _apply_last_axis(self.down, x)) * self.residual_scale

    def delta_weight(self) -> jax.Array:
        """Return the dense linear branch update."""

        return (self.up.weight @ self.down.weight) * self.residual_scale

    def delta_bias(self) -> jax.Array | None:
        """Return the bias contribution of the reparameterized branch."""

        down_bias = self.down.bias
        up_bias = self.up.bias
        if down_bias is None and up_bias is None:
            return None
        dtype = self.up.weight.dtype
        bias = jnp.zeros((self.up.out_features,), dtype=dtype)
        if down_bias is not None:
            bias = bias + self.up.weight @ down_bias
        if up_bias is not None:
            bias = bias + up_bias
        return bias * self.residual_scale

    def merge(self):
        """Return a module with the RepAdapter branch folded into ``base``."""

        if not self.mergeable:
            raise ValueError("This RepAdapter module is not mergeable.")
        if self.merged:
            raise ValueError("RepAdapter module is already merged.")
        base = eqx.tree_at(
            lambda linear: linear.weight,
            self.base,
            _linear_weight(self.base) + self.delta_weight().astype(_linear_weight(self.base).dtype),
        )
        delta_bias = self.delta_bias()
        if delta_bias is not None:
            base_bias = _linear_bias(base)
            if base_bias is None:
                raise ValueError("Cannot merge a biased RepAdapter into a biasless base.")
            base = eqx.tree_at(
                lambda linear: linear.bias,
                base,
                base_bias + delta_bias.astype(base_bias.dtype),
            )
        return self._replace(base=base, merged=True)

    def unmerge(self):
        """Return a module with the RepAdapter branch removed from ``base``."""

        if not self.merged:
            return self
        base = eqx.tree_at(
            lambda linear: linear.weight,
            self.base,
            _linear_weight(self.base) - self.delta_weight().astype(_linear_weight(self.base).dtype),
        )
        delta_bias = self.delta_bias()
        if delta_bias is not None:
            base_bias = _linear_bias(base)
            if base_bias is None:
                raise ValueError("Cannot unmerge a biased RepAdapter from a biasless base.")
            base = eqx.tree_at(
                lambda linear: linear.bias,
                base,
                base_bias - delta_bias.astype(base_bias.dtype),
            )
        return self._replace(base=base, merged=False)

    def _replace(self, *, base: eqx.Module, merged: bool):
        return RepAdapterLinear(
            base,
            bottleneck=int(self.down.out_features),
            residual_scale_init=float(self.residual_scale),
            down_init="kaiming_uniform",
            up_init="zeros",
            branch_bias=self.down.bias is not None or self.up.bias is not None,
            train_base=self.train_base,
            mergeable=self.mergeable,
            key=jr.PRNGKey(0),
            down=self.down,
            up=self.up,
            residual_scale=self.residual_scale,
            merged=merged,
        )


class ConvPassAdapter(eqx.Module):
    """Vision-native ConvPass branch for token grids."""

    down: eqx.nn.Linear
    up: eqx.nn.Linear
    spatial_kernel: jax.Array
    patch_grid: tuple[int, int] = eqx.field(static=True)
    activation: ActivationName = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    class_token_policy: Literal["separate", "bypass", "exclude"] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        bottleneck: int,
        *,
        patch_grid: tuple[int, int],
        key: jax.Array,
        spatial_kernel: int = 3,
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
        up_init: str = "zeros",
        class_token_policy: Literal["separate", "bypass", "exclude"] = "separate",
        down: eqx.nn.Linear | None = None,
        up: eqx.nn.Linear | None = None,
        kernel: jax.Array | None = None,
    ):
        if spatial_kernel < 1 or spatial_kernel % 2 == 0:
            raise ValueError("ConvPass spatial_kernel must be a positive odd integer.")
        key_down, key_up, key_conv = jr.split(key, 3)
        self.down = (
            _init_linear(eqx.nn.Linear(dim, bottleneck, key=key_down), key_down, "kaiming_uniform")
            if down is None
            else down
        )
        self.up = (
            _init_linear(eqx.nn.Linear(bottleneck, dim, key=key_up), key_up, up_init)
            if up is None
            else up
        )
        fan = spatial_kernel * spatial_kernel
        self.spatial_kernel = (
            jr.normal(
                key_conv,
                (bottleneck, spatial_kernel, spatial_kernel),
                dtype=self.down.weight.dtype,
            )
            / jnp.sqrt(fan)
            if kernel is None
            else kernel
        )
        self.patch_grid = patch_grid
        self.activation = activation
        self.dropout = dropout
        self.class_token_policy = class_token_policy

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        if x.ndim == 3:
            keys = _split_optional_key(key, x.shape[0])
            return jnp.stack(
                [
                    self(tokens, key=token_key, inference=inference)
                    for tokens, token_key in zip(x, keys, strict=True)
                ]
            )
        if x.ndim != 2:
            raise ValueError("ConvPassAdapter expects token arrays with shape (tokens, dim).")
        height, width = self.patch_grid
        patch_count = height * width
        if x.shape[0] < patch_count:
            raise ValueError("ConvPass patch_grid is larger than the token sequence.")
        prefix_count = x.shape[0] - patch_count
        prefix = x[:prefix_count]
        patches = x[prefix_count:]
        z = _apply_last_axis(self.down, patches)
        z = z.reshape(height, width, z.shape[-1])
        z = _depthwise_conv2d(z, self.spatial_kernel)
        z = _activation(self.activation)(z)
        if self.dropout > 0.0 and not inference:
            if key is None:
                raise ValueError("A PRNG key is required when ConvPass dropout is active.")
            z = _dropout(z, self.dropout, key)
        patch_delta = _apply_last_axis(self.up, z.reshape(patch_count, z.shape[-1]))
        if prefix_count == 0:
            return patch_delta
        prefix_delta = jnp.zeros_like(prefix)
        if self.class_token_policy == "exclude":
            return jnp.concatenate([prefix_delta, patch_delta], axis=0)
        if self.class_token_policy in {"separate", "bypass"}:
            return jnp.concatenate([prefix_delta, patch_delta], axis=0)
        raise ValueError(f"Unsupported ConvPass class_token_policy {self.class_token_policy!r}.")


class ConvPassBlock(eqx.Module):
    """Wrap a ViT block with one or more ConvPass residual branches."""

    base: eqx.Module
    convpass: ConvPassAdapter
    placements: tuple[Literal["attention", "mlp"], ...] = eqx.field(static=True)
    train_base: bool = eqx.field(static=True, default=False)

    def __call__(self, x: jax.Array, *args, key: jax.Array | None = None, inference: bool | None = None, **kwargs):
        key_base, key_adapter = _split_pair(key)
        y = _call_base(self.base, x, *args, key=key_base, inference=inference, **kwargs)
        return y + self.convpass(x, key=key_adapter, inference=inference)


def apply_adapters(
    model: PyTree,
    config: AdapterConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Insert bottleneck adapters into selected blocks."""

    config = AdapterConfig() if config is None else config
    block_paths = _target_block_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(block_paths))
    updated = model

    for block_path, subkey in zip(block_paths, keys, strict=True):
        block = get_path(updated, block_path)
        if isinstance(block, (SerialAdapterBlock, ParallelAdapterBlock, AdaptFormerBlock)):
            continue
        dim = _infer_dim(block)
        bottleneck = _resolve_bottleneck(dim, config)
        if config.placement == "parallel":
            adapter_keys = jr.split(subkey, 1)
            wrapper = ParallelAdapterBlock(
                block,
                BottleneckAdapter(
                    dim,
                    bottleneck,
                    key=adapter_keys[0],
                    activation=config.activation,
                    dropout=config.dropout,
                    down_init=config.down_init,
                    up_init=config.up_init,
                    residual_scale_init=config.residual_scale_init,
                    pre_norm=config.pre_norm,
                ),
                train_base=config.train_base,
            )
            updated = eqx.tree_at(lambda tree, p=block_path: get_path(tree, p), updated, wrapper)
        else:
            site_paths = _adapter_site_paths(block, block_path, config.placement)
            for site_path, adapter_key in zip(
                site_paths,
                jr.split(subkey, len(site_paths)),
                strict=True,
            ):
                updated = _add_output_adapter_at_path(
                    updated,
                    site_path,
                    bottleneck=bottleneck,
                    name=None,
                    key=adapter_key,
                    config=config,
                )

    return updated


def add_adapter(
    model: PyTree,
    *,
    name: str,
    config: AdapterConfig | None = None,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Add a named serial adapter bank entry and keep the current active entry."""

    if not name:
        raise ValueError("Adapter name must be a non-empty string.")
    config = AdapterConfig() if config is None else config
    if config.placement == "parallel":
        raise ValueError("Named adapter banks currently support 'after_mlp' and 'both'.")

    block_paths = _target_block_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(block_paths))
    updated = model
    for block_path, subkey in zip(block_paths, keys, strict=True):
        block = get_path(updated, block_path)
        if isinstance(block, ParallelAdapterBlock):
            raise ValueError("Named adapter banks do not support parallel adapter wrappers.")
        if isinstance(block, SerialAdapterBlock) and block.adapter_fusion is not None:
            raise ValueError("Add adapters before applying AdapterFusion.")

        base = block.base if isinstance(block, SerialAdapterBlock) else block
        dim = _infer_dim(base)
        bottleneck = _resolve_bottleneck(dim, config)
        site_paths = _adapter_site_paths(base, block_path, config.placement)
        for site_path, adapter_key in zip(
            site_paths,
            jr.split(subkey, len(site_paths)),
            strict=True,
        ):
            existing = get_path(updated, site_path)
            if isinstance(existing, OutputAdapterModule) and existing.adapter_fusion is not None:
                raise ValueError("Add adapters before applying AdapterFusion.")
            existing_names = _wrapper_adapter_names(existing)
            if name in existing_names:
                raise ValueError(f"Adapter {name!r} already exists at {path_to_str(site_path)}.")
            updated = _add_output_adapter_at_path(
                updated,
                site_path,
                bottleneck=bottleneck,
                name=name,
                key=adapter_key,
                config=config,
            )
    return updated


def set_active_adapter(model: PyTree, name: str) -> PyTree:
    """Return ``model`` with all named serial adapter banks switched to ``name``."""

    return configure_adapter_bank(model, AdapterBankConfig(active=name))


def configure_adapter_bank(
    model: PyTree,
    config: AdapterBankConfig,
) -> PyTree:
    """Return ``model`` with named adapter banks configured per ``config``."""

    active = _normalize_active_adapter(config.active)
    if len(active) > 1 and not config.allow_multiple_active:
        raise ValueError("Multiple active adapters require allow_multiple_active=True.")

    wrappers = iter_adapter_wrappers(model)
    named_wrappers = [
        (path, wrapper)
        for path, wrapper in wrappers
        if isinstance(wrapper, (SerialAdapterBlock, OutputAdapterModule))
        and wrapper.adapter_names
    ]
    if not named_wrappers:
        if config.missing_adapter_policy == "ignore":
            return model
        raise ValueError("No named adapter banks found.")

    updated = model
    for path, wrapper in named_wrappers:
        missing = tuple(name for name in active if name not in wrapper.adapter_names)
        if missing:
            if config.missing_adapter_policy == "ignore":
                continue
            raise ValueError(f"Adapter {missing[0]!r} not found at {path_to_str(path)}.")
        updated_wrapper = wrapper.__class__(
            wrapper.base,
            wrapper.adapters,
            wrapper.adapter_names,
            active[0] if len(active) == 1 else active,
            wrapper.adapter_fusion,
            train_base=wrapper.train_base,
        )
        updated = eqx.tree_at(
            lambda tree, p=path: get_path(tree, p),
            updated,
            updated_wrapper,
        )
    return updated


def apply_adapter_fusion(
    model: PyTree,
    config: AdapterFusionConfig | None = None,
    *,
    key: jax.Array,
) -> PyTree:
    """Attach attention fusion modules to named serial adapter banks."""

    config = AdapterFusionConfig() if config is None else config
    if config.fusion != "attention":
        raise ValueError("AdapterFusion currently supports fusion='attention'.")
    wrappers = [
        (path, wrapper)
        for path, wrapper in iter_adapter_wrappers(model)
        if isinstance(wrapper, (SerialAdapterBlock, OutputAdapterModule))
        and wrapper.adapter_names
        and _adapter_fusion_matches_placement(path, config.placement)
    ]
    if not wrappers:
        raise ValueError(
            "AdapterFusion requires named serial adapter banks at the configured "
            "placement."
        )
    keys = jr.split(key, len(wrappers))
    updated = model
    for (path, wrapper), subkey in zip(wrappers, keys, strict=True):
        active = _fusion_active_names(wrapper)
        adapters = _adapters_for_names(wrapper, active)
        dim = _fusion_dim(wrapper)
        updated_wrapper = wrapper.__class__(
            wrapper.base,
            wrapper.adapters,
            wrapper.adapter_names,
            active[0] if len(active) == 1 else active,
            AdapterFusion(
                dim,
                len(adapters),
                key=subkey,
                fusion=config.fusion,
                dropout=config.fusion_dropout,
            ),
            train_base=wrapper.train_base,
        )
        updated = eqx.tree_at(
            lambda tree, p=path: get_path(tree, p),
            updated,
            updated_wrapper,
        )
    return updated


def adapter_fusion_trainable_spec(
    config: AdapterFusionConfig | None = None,
    *,
    train_head: bool = True,
) -> TrainableSpec:
    """Return the trainability mask for AdapterFusion training."""

    config = AdapterFusionConfig() if config is None else config
    tags = ("adapter_fusion",) if config.freeze_task_adapters else ("adapter",)
    return TrainableSpec(
        mode="peft",
        target=TargetSpec(tags_any=tags),
        train_head=train_head,
    )


def apply_adaptformer(
    model: PyTree,
    config: AdaptFormerConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Insert AdaptFormer-style adapters into selected blocks."""

    config = AdaptFormerConfig() if config is None else config
    block_paths = _target_block_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(block_paths))
    updated = model

    for block_path, subkey in zip(block_paths, keys, strict=True):
        block = get_path(updated, block_path)
        if isinstance(block, AdaptFormerBlock):
            continue
        dim = _infer_dim(block)
        wrapper = AdaptFormerBlock(
            block,
            AdaptFormerAdapter(
                dim,
                config.bottleneck,
                key=subkey,
                activation=config.activation,
                dropout=config.dropout,
                up_init=config.up_init,
                scale_init=config.scale_init,
            ),
        )
        updated = eqx.tree_at(lambda tree, p=block_path: get_path(tree, p), updated, wrapper)

    return updated


def apply_orthogonal_adapters(
    model: PyTree,
    config: OrthogonalAdapterConfig | None = None,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply OFT/BOFT-style orthogonal wrappers to selected linears."""

    config = OrthogonalAdapterConfig() if config is None else config
    module_paths = _target_linear_module_paths(model, config.target, tagger=tagger)
    updated = model
    for module_path in module_paths:
        module = get_path(updated, module_path)
        if isinstance(module, OrthogonalLinear):
            continue
        if not _is_linear_like(module):
            raise TypeError(
                f"Orthogonal adapter target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected a linear-like module."
            )
        updated = eqx.tree_at(
            lambda tree, p=module_path: get_path(tree, p),
            updated,
            OrthogonalLinear(
                module,
                side=config.side,
                parameterization=config.parameterization,
                block_size=config.block_size,
                num_factors=config.num_factors,
                eps=config.eps,
                train_base=config.train_base,
                mergeable=config.mergeable,
            ),
        )
    return updated


def apply_repadapter(
    model: PyTree,
    config: RepAdapterConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply structurally mergeable RepAdapter branches to selected linears."""

    config = RepAdapterConfig() if config is None else config
    module_paths = _target_linear_module_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(module_paths))
    updated = model
    for module_path, subkey in zip(module_paths, keys, strict=True):
        module = get_path(updated, module_path)
        if isinstance(module, RepAdapterLinear):
            continue
        if not _is_linear_like(module):
            raise TypeError(
                f"RepAdapter target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected a linear-like module."
            )
        updated = eqx.tree_at(
            lambda tree, p=module_path: get_path(tree, p),
            updated,
            RepAdapterLinear(
                module,
                bottleneck=config.bottleneck,
                residual_scale_init=config.residual_scale_init,
                down_init=config.down_init,
                up_init=config.up_init,
                branch_bias=config.branch_bias,
                train_base=config.train_base,
                mergeable=config.mergeable,
                key=subkey,
            ),
        )
    return updated


def apply_convpass(
    model: PyTree,
    config: ConvPassConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Insert ConvPass branches into selected vision transformer blocks."""

    config = ConvPassConfig() if config is None else config
    patch_grid = _resolve_patch_grid(model, config.patch_grid)
    block_paths = _target_block_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(block_paths))
    updated = model
    for block_path, subkey in zip(block_paths, keys, strict=True):
        block = get_path(updated, block_path)
        if isinstance(block, ConvPassBlock):
            continue
        dim = _infer_dim(block)
        updated = eqx.tree_at(
            lambda tree, p=block_path: get_path(tree, p),
            updated,
            ConvPassBlock(
                block,
                ConvPassAdapter(
                    dim,
                    config.bottleneck,
                    patch_grid=patch_grid,
                    key=subkey,
                    spatial_kernel=config.spatial_kernel,
                    activation=config.activation,
                    dropout=config.dropout,
                    up_init=config.up_init,
                    class_token_policy=config.class_token_policy,
                ),
                config.placement,
                train_base=config.train_base,
            ),
        )
    return updated


def merge_orthogonal_adapters(model: PyTree) -> PyTree:
    """Merge every orthogonal adapter in ``model``."""

    updated = model
    for path, module in iter_orthogonal_adapters(updated):
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, module.merge())
    return updated


def merge_repadapters(model: PyTree) -> PyTree:
    """Merge every RepAdapter module in ``model``."""

    updated = model
    for path, module in iter_repadapters(updated):
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, module.merge())
    return updated


def unmerge_orthogonal_adapters(model: PyTree) -> PyTree:
    """Unmerge every orthogonal adapter in ``model``."""

    updated = model
    for path, module in iter_orthogonal_adapters(updated):
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, module.unmerge())
    return updated


def unmerge_repadapters(model: PyTree) -> PyTree:
    """Unmerge every merged RepAdapter module in ``model``."""

    updated = model
    for path, module in iter_repadapters(updated):
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, module.unmerge())
    return updated


def extract_adapter_delta(model: PyTree) -> FineTuneBundle:
    """Extract adapter wrapper state into a delta bundle."""

    entries = []
    for path, wrapper in iter_adapter_wrappers(model):
        entries.append(_adapter_entry(path, wrapper))
    for path, wrapper in iter_orthogonal_adapters(model):
        entries.append(_orthogonal_entry(path, wrapper))
    for path, wrapper in iter_repadapters(model):
        entries.append(_repadapter_entry(path, wrapper))
    entries.sort(key=lambda entry: entry["path"])

    return FineTuneBundle(
        method="adapter",
        schema_version=1,
        architecture_hash=architecture_hash(strip_adapters(model)),
        adapter_config={"entries": entries},
    )


def load_adapter_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    """Load adapter deltas into a compatible base model."""

    if bundle.method != "adapter":
        raise FineTuneBundleError(
            f"Expected an adapter bundle, got method={bundle.method!r}."
        )
    actual_hash = architecture_hash(base_model)
    if bundle.architecture_hash and bundle.architecture_hash != actual_hash:
        raise FineTuneBundleError(
            "Adapter delta architecture hash mismatch: "
            f"expected {bundle.architecture_hash}, got {actual_hash}."
        )

    updated = base_model
    for entry in bundle.adapter_config.get("entries", ()):
        path = str_to_path(entry["path"])
        block = _bundle_get_path(updated, path, method_name="Adapter")
        wrapper = _wrapper_from_entry(block, entry)
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, wrapper)
    return updated


def strip_adapters(model: PyTree) -> PyTree:
    """Replace adapter wrappers by their base blocks."""

    stripped = model
    for path, wrapper in iter_adapter_wrappers(stripped):
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, wrapper.base)
    for path, wrapper in iter_orthogonal_adapters(stripped):
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, wrapper.base)
    for path, wrapper in iter_repadapters(stripped):
        base = wrapper.unmerge().base
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, base)
    return stripped


def iter_adapter_wrappers(
    model: PyTree,
) -> tuple[
    tuple[
        Path,
        SerialAdapterBlock | OutputAdapterModule | ParallelAdapterBlock | AdaptFormerBlock | ConvPassBlock,
    ],
    ...,
]:
    """Return path/wrapper pairs for adapter-wrapped blocks."""

    wrapper_types = (
        SerialAdapterBlock,
        OutputAdapterModule,
        ParallelAdapterBlock,
        AdaptFormerBlock,
        ConvPassBlock,
    )
    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, wrapper_types),
        )
        if isinstance(leaf, wrapper_types)
    )


def iter_orthogonal_adapters(model: PyTree) -> tuple[tuple[Path, OrthogonalLinear], ...]:
    """Return path/module pairs for orthogonal linear wrappers in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, OrthogonalLinear),
        )
        if isinstance(leaf, OrthogonalLinear)
    )


def iter_repadapters(model: PyTree) -> tuple[tuple[Path, RepAdapterLinear], ...]:
    """Return path/module pairs for RepAdapter wrappers in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, RepAdapterLinear),
        )
        if isinstance(leaf, RepAdapterLinear)
    )


def _adapter_entry(
    path: Path,
    wrapper: SerialAdapterBlock | OutputAdapterModule | ParallelAdapterBlock | AdaptFormerBlock | ConvPassBlock,
) -> dict[str, Any]:
    if isinstance(wrapper, (SerialAdapterBlock, OutputAdapterModule)):
        return {
            "path": path_to_str(path),
            "class": wrapper.__class__.__name__,
            "adapters": [_bottleneck_state(adapter) for adapter in wrapper.adapters],
            "adapter_names": wrapper.adapter_names,
            "active_adapter": wrapper.active_adapter,
            "adapter_fusion": None
            if wrapper.adapter_fusion is None
            else _adapter_fusion_state(wrapper.adapter_fusion),
            "train_base": wrapper.train_base,
        }
    if isinstance(wrapper, ParallelAdapterBlock):
        return {
            "path": path_to_str(path),
            "class": "ParallelAdapterBlock",
            "adapter": _bottleneck_state(wrapper.adapter),
            "train_base": wrapper.train_base,
        }
    if isinstance(wrapper, ConvPassBlock):
        return {
            "path": path_to_str(path),
            "class": "ConvPassBlock",
            "convpass": _convpass_state(wrapper.convpass),
            "placements": wrapper.placements,
            "train_base": wrapper.train_base,
        }
    return {
        "path": path_to_str(path),
        "class": "AdaptFormerBlock",
        "adapter": _adaptformer_state(wrapper.adapter),
    }


def _orthogonal_entry(path: Path, wrapper: OrthogonalLinear) -> dict[str, Any]:
    return {
        "path": path_to_str(path),
        "class": "OrthogonalLinear",
        "orthogonal": _orthogonal_state(wrapper),
        "metadata": _orthogonal_serialization_metadata(wrapper),
    }


def _repadapter_entry(path: Path, wrapper: RepAdapterLinear) -> dict[str, Any]:
    return {
        "path": path_to_str(path),
        "class": "RepAdapterLinear",
        "repadapter": _repadapter_state(wrapper),
    }


def _wrapper_from_entry(block: eqx.Module, entry: dict[str, Any]):
    if entry["class"] == "SerialAdapterBlock":
        return SerialAdapterBlock(
            block,
            tuple(_bottleneck_from_state(state) for state in entry["adapters"]),
            tuple(entry.get("adapter_names", ())),
            _entry_active_adapter(entry.get("active_adapter")),
            None
            if entry.get("adapter_fusion") is None
            else _adapter_fusion_from_state(entry["adapter_fusion"]),
            train_base=bool(entry.get("train_base", False)),
        )
    if entry["class"] == "OutputAdapterModule":
        return OutputAdapterModule(
            block,
            tuple(_bottleneck_from_state(state) for state in entry["adapters"]),
            tuple(entry.get("adapter_names", ())),
            _entry_active_adapter(entry.get("active_adapter")),
            None
            if entry.get("adapter_fusion") is None
            else _adapter_fusion_from_state(entry["adapter_fusion"]),
            train_base=bool(entry.get("train_base", False)),
        )
    if entry["class"] == "ParallelAdapterBlock":
        return ParallelAdapterBlock(
            block,
            _bottleneck_from_state(entry["adapter"]),
            train_base=bool(entry.get("train_base", False)),
        )
    if entry["class"] == "AdaptFormerBlock":
        return AdaptFormerBlock(block, _adaptformer_from_state(entry["adapter"]))
    if entry["class"] == "ConvPassBlock":
        return ConvPassBlock(
            block,
            _convpass_from_state(entry["convpass"]),
            tuple(entry.get("placements", ("attention", "mlp"))),
            train_base=bool(entry.get("train_base", False)),
        )
    if entry["class"] == "OrthogonalLinear":
        return _orthogonal_from_state(block, entry["orthogonal"])
    if entry["class"] == "RepAdapterLinear":
        return _repadapter_from_state(block, entry["repadapter"])
    raise FineTuneBundleError(f"Unknown adapter wrapper class {entry['class']!r}.")


def _bundle_get_path(model: PyTree, path: Path, *, method_name: str):
    try:
        return get_path(model, path)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise FineTuneBundleError(
            f"{method_name} delta expects path {path_to_str(path)}, "
            "but the base model has no matching leaf."
        ) from error


def _bottleneck_state(adapter: BottleneckAdapter) -> dict[str, Any]:
    return {
        "dim": adapter.down.in_features,
        "bottleneck": adapter.down.out_features,
        "activation": adapter.activation,
        "dropout": adapter.dropout,
        "residual_scale": adapter.residual_scale,
        "norm": None if adapter.norm is None else _layer_norm_state(adapter.norm),
        "down": _linear_state(adapter.down),
        "up": _linear_state(adapter.up),
    }


def _adaptformer_state(adapter: AdaptFormerAdapter) -> dict[str, Any]:
    state = _bottleneck_state(adapter.adapter)
    state["scale"] = adapter.scale
    return state


def _convpass_state(adapter: ConvPassAdapter) -> dict[str, Any]:
    return {
        "dim": adapter.down.in_features,
        "bottleneck": adapter.down.out_features,
        "patch_grid": adapter.patch_grid,
        "activation": adapter.activation,
        "dropout": adapter.dropout,
        "class_token_policy": adapter.class_token_policy,
        "down": _linear_state(adapter.down),
        "up": _linear_state(adapter.up),
        "spatial_kernel": adapter.spatial_kernel,
    }


def _orthogonal_state(adapter: OrthogonalLinear) -> dict[str, Any]:
    return {
        "side": adapter.side,
        "parameterization": adapter.parameterization,
        "block_size": adapter.block_size,
        "num_factors": adapter.num_factors,
        "eps": adapter.eps,
        "train_base": adapter.train_base,
        "mergeable": adapter.mergeable,
        "merged": adapter.merged,
        "skew": adapter.skew,
        "serialization": _orthogonal_serialization_metadata(adapter),
    }


def _repadapter_state(adapter: RepAdapterLinear) -> dict[str, Any]:
    return {
        "bottleneck": adapter.down.out_features,
        "residual_scale": adapter.residual_scale,
        "branch_bias": adapter.down.bias is not None or adapter.up.bias is not None,
        "train_base": adapter.train_base,
        "mergeable": adapter.mergeable,
        "merged": adapter.merged,
        "down": _linear_state(adapter.down),
        "up": _linear_state(adapter.up),
        "serialization": {
            "layout": "linear_branch",
            "merge_rule": "W <- W + scale * W_up @ W_down",
            "bias_rule": "b <- b + scale * (W_up @ b_down + b_up)",
        },
    }


def _orthogonal_serialization_metadata(adapter: OrthogonalLinear) -> dict[str, Any]:
    dim = _linear_weight(adapter.base).shape[1] if adapter.side == "input" else _linear_weight(adapter.base).shape[0]
    block_size = adapter.block_size
    num_blocks = 1 if block_size is None else dim // block_size
    factor_order = tuple(range(adapter.num_factors))
    factor_permutations = (
        (tuple(range(dim)),)
        if block_size is None
        else tuple(
            _boft_factor_permutation(dim, block_size, factor_index)
            for factor_index in factor_order
        )
    )
    return {
        "parameterization": adapter.parameterization,
        "layout": "dense_cayley" if block_size is None else "sparse_butterfly_cayley",
        "adapted_dimension": dim,
        "block_size": block_size,
        "block_order": tuple(range(num_blocks)),
        "factor_order": factor_order,
        "butterfly_permutation": factor_permutations,
    }


def _adapter_fusion_state(fusion: AdapterFusion) -> dict[str, Any]:
    return {
        "dim": fusion.scorer.in_features,
        "num_adapters": fusion.scorer.out_features,
        "fusion": fusion.fusion,
        "dropout": fusion.dropout,
        "scorer": _linear_state(fusion.scorer),
    }


def _bottleneck_from_state(state: dict[str, Any]) -> BottleneckAdapter:
    return BottleneckAdapter(
        int(state["dim"]),
        int(state["bottleneck"]),
        key=jr.PRNGKey(0),
        activation=state["activation"],
        dropout=float(state["dropout"]),
        norm=(
            None
            if state.get("norm") is None
            else _layer_norm_from_state(state["norm"])
        ),
        down=_linear_from_state(state["down"]),
        up=_linear_from_state(state["up"]),
        residual_scale=state.get("residual_scale"),
    )


def _adaptformer_from_state(state: dict[str, Any]) -> AdaptFormerAdapter:
    adapter = _bottleneck_from_state(state)
    return AdaptFormerAdapter(
        int(state["dim"]),
        int(state["bottleneck"]),
        key=jr.PRNGKey(0),
        activation=state["activation"],
        dropout=float(state["dropout"]),
        adapter=adapter,
        scale=state["scale"],
    )


def _convpass_from_state(state: dict[str, Any]) -> ConvPassAdapter:
    return ConvPassAdapter(
        int(state["dim"]),
        int(state["bottleneck"]),
        patch_grid=tuple(state["patch_grid"]),
        key=jr.PRNGKey(0),
        spatial_kernel=int(state["spatial_kernel"].shape[-1]),
        activation=state["activation"],
        dropout=float(state["dropout"]),
        class_token_policy=state["class_token_policy"],
        down=_linear_from_state(state["down"]),
        up=_linear_from_state(state["up"]),
        kernel=state["spatial_kernel"],
    )


def _orthogonal_from_state(base: eqx.Module, state: dict[str, Any]) -> OrthogonalLinear:
    return OrthogonalLinear(
        base,
        side=state.get("side", "input"),
        parameterization=state.get("parameterization", "cayley"),
        block_size=state.get("block_size"),
        num_factors=int(state.get("num_factors", 1)),
        eps=float(state.get("eps", 1e-6)),
        train_base=bool(state.get("train_base", False)),
        mergeable=bool(state.get("mergeable", True)),
        skew=state["skew"],
        merged=bool(state.get("merged", False)),
    )


def _repadapter_from_state(base: eqx.Module, state: dict[str, Any]) -> RepAdapterLinear:
    adapter = RepAdapterLinear(
        base,
        bottleneck=int(state["bottleneck"]),
        residual_scale_init=1.0,
        down_init="kaiming_uniform",
        up_init="zeros",
        branch_bias=bool(state.get("branch_bias", False)),
        train_base=bool(state.get("train_base", False)),
        mergeable=bool(state.get("mergeable", True)),
        key=jr.PRNGKey(0),
        down=_linear_from_state(state["down"]),
        up=_linear_from_state(state["up"]),
        residual_scale=state["residual_scale"],
        merged=False,
    )
    if state.get("merged", False):
        adapter = adapter.merge()
    return adapter


def _adapter_fusion_from_state(state: dict[str, Any]) -> AdapterFusion:
    return AdapterFusion(
        int(state["dim"]),
        int(state["num_adapters"]),
        key=jr.PRNGKey(0),
        fusion=state["fusion"],
        dropout=float(state["dropout"]),
        scorer=_linear_from_state(state["scorer"]),
    )


def _adapter_fusion_matches_placement(path: Path, placements: tuple[str, ...]) -> bool:
    parts = {str(part) for part in path}
    for placement in placements:
        if placement == "after_mlp" and parts.intersection(
            {"mlp", "ffn", "feed_forward", "feedforward"}
        ):
            return True
        if placement in {"after_attention", "after_attn"} and parts.intersection(
            {"attn", "attention", "self_attn", "self_attention"}
        ):
            return True
    return False


def _target_linear_module_paths(
    model: PyTree,
    target: TargetSpec,
    *,
    tagger: Tagger,
) -> tuple[Path, ...]:
    paths = {_linear_module_path(info.path) for info in resolve_target(model, target, tagger=tagger)}
    if not paths and not target.allow_empty:
        raise ValueError("TargetSpec resolved no linear modules.")
    return tuple(sorted(paths, key=path_to_str))


def _linear_module_path(path: Path) -> Path:
    if path[-1:] in (("weight",), ("bias",)):
        return path[:-1]
    return path


def _is_linear_like(module: Any) -> bool:
    weight = getattr(module, "weight", None)
    return callable(module) and eqx.is_inexact_array(weight) and weight.ndim == 2


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


def _linear_call(weight: jax.Array, bias: jax.Array | None, x: jax.Array) -> jax.Array:
    if x.ndim == 1:
        y = weight @ x
        return y if bias is None else y + bias
    y = x @ weight.T
    return y if bias is None else y + bias


def _orthogonal_skew_shape(
    dim: int,
    block_size: int | None,
    *,
    num_factors: int = 1,
) -> tuple[int, ...]:
    if block_size is None:
        return (dim, dim)
    block_shape = (dim // block_size, block_size, block_size)
    return block_shape if num_factors == 1 else (num_factors, *block_shape)


def _orthogonal_from_skew(
    skew_param: jax.Array,
    *,
    eps: float,
    block_size: int | None,
) -> jax.Array:
    if block_size is None:
        return _cayley(skew_param, eps=eps)
    dim = _boft_adapted_dim(skew_param)
    basis_rows = _apply_boft_to_last_axis(
        jnp.eye(dim, dtype=skew_param.dtype),
        skew_param,
        eps=eps,
    )
    return basis_rows.T


def _boft_factor_skews(skew_param: jax.Array) -> jax.Array:
    if skew_param.ndim == 3:
        return skew_param[None, ...]
    if skew_param.ndim == 4:
        return skew_param
    raise ValueError("BOFT skew state must have shape (blocks, b, b) or (factors, blocks, b, b).")


def _boft_adapted_dim(skew_param: jax.Array) -> int:
    factors = _boft_factor_skews(skew_param)
    return int(factors.shape[-3] * factors.shape[-1])


def _boft_factor_permutation(
    dim: int,
    block_size: int,
    factor_index: int,
) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    while stride * block_size <= dim:
        if dim % (stride * block_size) == 0:
            strides.append(stride)
        stride *= block_size
    if not strides:
        strides = [1]
    stride = strides[factor_index % len(strides)]
    permutation: list[int] = []
    span = stride * block_size
    for window_start in range(0, dim, span):
        for offset in range(stride):
            permutation.extend(window_start + offset + step * stride for step in range(block_size))
    return tuple(permutation)


def _apply_boft_to_last_axis(
    x: jax.Array,
    skew_param: jax.Array,
    *,
    eps: float,
) -> jax.Array:
    factors = _boft_factor_skews(skew_param)
    y = x
    for factor_index in range(factors.shape[0]):
        y = _apply_boft_factor_to_last_axis(
            y,
            factors[factor_index],
            factor_index=factor_index,
            eps=eps,
        )
    return y


def _apply_boft_factor_to_last_axis(
    x: jax.Array,
    skew_param: jax.Array,
    *,
    factor_index: int,
    eps: float,
) -> jax.Array:
    blocks = jax.vmap(lambda block: _cayley(block, eps=eps))(skew_param)
    block_size = blocks.shape[-1]
    leading = x.shape[:-1]
    if x.shape[-1] % block_size != 0:
        raise ValueError("BOFT block_size must divide the feature axis.")
    permutation = jnp.asarray(
        _boft_factor_permutation(int(x.shape[-1]), int(block_size), factor_index),
        dtype=jnp.int32,
    )
    inverse = jnp.argsort(permutation)
    x_perm = jnp.take(x, permutation, axis=-1)
    x_blocks = x_perm.reshape((*leading, x.shape[-1] // block_size, block_size))
    y_blocks = jnp.einsum("...nb,nab->...na", x_blocks, blocks)
    y_perm = y_blocks.reshape(x.shape)
    return jnp.take(y_perm, inverse, axis=-1)


def _apply_orthogonal_blocks_to_last_axis(
    x: jax.Array,
    skew_param: jax.Array,
    *,
    eps: float,
) -> jax.Array:
    blocks = jax.vmap(lambda block: _cayley(block, eps=eps))(skew_param)
    block_size = blocks.shape[-1]
    leading = x.shape[:-1]
    if x.shape[-1] % block_size != 0:
        raise ValueError("BOFT block_size must divide the feature axis.")
    x_blocks = x.reshape((*leading, x.shape[-1] // block_size, block_size))
    y_blocks = jnp.einsum("...nb,nab->...na", x_blocks, blocks)
    return y_blocks.reshape(x.shape)


def _cayley(skew_param: jax.Array, *, eps: float) -> jax.Array:
    del eps
    skew = skew_param - skew_param.T
    dim = skew.shape[0]
    eye = jnp.eye(dim, dtype=skew.dtype)
    return jnp.linalg.solve(eye + skew, eye - skew)


def _depthwise_conv2d(tokens_hwc: jax.Array, kernel: jax.Array) -> jax.Array:
    channels, kernel_h, kernel_w = kernel.shape
    if tokens_hwc.shape[-1] != channels:
        raise ValueError("ConvPass kernel channel count does not match token features.")
    padding = ((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2))
    x = jnp.moveaxis(tokens_hwc, -1, 0)[None, :, :, :]
    w = kernel[:, None, :, :]
    y = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=channels,
    )
    return jnp.moveaxis(y[0], 0, -1)


def _resolve_patch_grid(model: PyTree, patch_grid: tuple[int, int] | None) -> tuple[int, int]:
    if patch_grid is not None:
        return patch_grid
    model_grid = getattr(model, "patch_grid", None)
    if model_grid is not None:
        return tuple(model_grid)
    raise ValueError("ConvPass requires explicit patch_grid metadata.")


def _layer_norm_state(norm: eqx.nn.LayerNorm) -> dict[str, Any]:
    return {
        "shape": norm.shape,
        "eps": norm.eps,
        "use_weight": norm.use_weight,
        "use_bias": norm.use_bias,
        "weight": norm.weight,
        "bias": norm.bias,
    }


def _layer_norm_from_state(state: dict[str, Any]) -> eqx.nn.LayerNorm:
    norm = eqx.nn.LayerNorm(
        tuple(state["shape"]),
        eps=float(state["eps"]),
        use_weight=bool(state["use_weight"]),
        use_bias=bool(state["use_bias"]),
    )
    if state.get("weight") is not None:
        norm = eqx.tree_at(lambda layer: layer.weight, norm, state["weight"])
    if state.get("bias") is not None:
        norm = eqx.tree_at(lambda layer: layer.bias, norm, state["bias"])
    return norm


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


def _target_block_paths(
    model: PyTree,
    target: TargetSpec,
    *,
    tagger: Tagger,
) -> tuple[Path, ...]:
    block_paths = {_block_path(info.path) for info in resolve_target(model, target, tagger=tagger)}
    block_paths.discard(())
    return tuple(sorted(block_paths, key=path_to_str))


def _block_path(path: Path) -> Path:
    parts = tuple(str(part) for part in path)
    for index, part in enumerate(parts[:-1]):
        if part in {"blocks", "block"}:
            return path[: index + 2]
    return path[:-1]


def _infer_dim(block: eqx.Module) -> int:
    for leaf in jtu.tree_leaves(block, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)):
        if isinstance(leaf, eqx.nn.Linear):
            return int(leaf.in_features)
    raise ValueError(f"Could not infer adapter feature dimension for {type(block).__name__}.")


def _adapter_site_paths(
    block: eqx.Module,
    block_path: Path,
    placement: AdapterPlacement,
) -> tuple[Path, ...]:
    if placement == "after_mlp":
        return (_child_path(block, block_path, ("mlp", "ffn", "feed_forward", "feedforward")),)
    if placement == "both":
        return (
            _child_path(
                block,
                block_path,
                ("attn", "attention", "self_attn", "self_attention"),
            ),
            _child_path(block, block_path, ("mlp", "ffn", "feed_forward", "feedforward")),
        )
    raise ValueError(f"Unsupported serial adapter placement {placement!r}.")


def _child_path(block: eqx.Module, block_path: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        if hasattr(block, name):
            return (*block_path, name)
    expected = ", ".join(names)
    raise ValueError(
        f"Adapter placement requires block {path_to_str(block_path)!r} to expose "
        f"one of: {expected}."
    )


def _add_output_adapter_at_path(
    model: PyTree,
    site_path: Path,
    *,
    bottleneck: int,
    name: str | None,
    key: jax.Array,
    config: AdapterConfig,
) -> PyTree:
    module = get_path(model, site_path)
    if isinstance(module, OutputAdapterModule):
        base = module.base
        existing_adapters = module.adapters
        existing_names = _wrapper_adapter_names(module)
        active = (
            module.active_adapter
            if module.active_adapter is not None
            else ("default" if existing_names else name)
        )
    else:
        base = module
        existing_adapters = ()
        existing_names = ()
        active = name

    dim = _infer_dim(base)
    adapter = BottleneckAdapter(
        dim,
        bottleneck,
        key=key,
        activation=config.activation,
        dropout=config.dropout,
        down_init=config.down_init,
        up_init=config.up_init,
        residual_scale_init=config.residual_scale_init,
        pre_norm=config.pre_norm,
    )
    names = existing_names
    if name is not None:
        names = (*existing_names, name)
    wrapper = OutputAdapterModule(
        base,
        (*existing_adapters, adapter),
        names,
        active,
        None,
        train_base=bool(config.train_base or getattr(module, "train_base", False)),
    )
    return eqx.tree_at(lambda tree, p=site_path: get_path(tree, p), model, wrapper)


def _wrapper_adapter_names(wrapper: Any) -> tuple[str, ...]:
    if not isinstance(wrapper, (SerialAdapterBlock, OutputAdapterModule)):
        return ()
    if wrapper.adapter_names:
        return wrapper.adapter_names
    return ("default",) * len(wrapper.adapters)


def _resolve_bottleneck(dim: int, config: AdapterConfig) -> int:
    if config.bottleneck is not None:
        return config.bottleneck
    bottleneck = max(1, dim // config.reduction_factor)
    clipped = min(max(bottleneck, config.bottleneck_min), config.bottleneck_max)
    return max(1, ((int(clipped) + 7) // 8) * 8)


def _active_adapters(
    wrapper: SerialAdapterBlock | OutputAdapterModule,
) -> tuple[BottleneckAdapter, ...]:
    if not wrapper.adapter_names or wrapper.active_adapter is None:
        return wrapper.adapters
    active = frozenset(_normalize_active_adapter(wrapper.active_adapter))
    return tuple(
        adapter
        for adapter, name in zip(wrapper.adapters, wrapper.adapter_names, strict=True)
        if name in active
    )


def _fusion_active_names(
    wrapper: SerialAdapterBlock | OutputAdapterModule,
) -> tuple[str, ...]:
    if wrapper.active_adapter is not None:
        active = _normalize_active_adapter(wrapper.active_adapter)
        if len(active) > 1:
            return active
    return tuple(dict.fromkeys(wrapper.adapter_names))


def _adapters_for_names(
    wrapper: SerialAdapterBlock | OutputAdapterModule,
    names: tuple[str, ...],
) -> tuple[BottleneckAdapter, ...]:
    active = frozenset(names)
    adapters = tuple(
        adapter
        for adapter, name in zip(wrapper.adapters, wrapper.adapter_names, strict=True)
        if name in active
    )
    if not adapters:
        raise ValueError("AdapterFusion requires at least one active adapter.")
    return adapters


def _fusion_dim(wrapper: SerialAdapterBlock | OutputAdapterModule) -> int:
    if wrapper.adapters:
        return int(wrapper.adapters[0].up.out_features)
    return _infer_dim(wrapper.base)


def _normalize_active_adapter(active: ActiveAdapter) -> tuple[str, ...]:
    if isinstance(active, str):
        if not active:
            raise ValueError("Adapter name must be a non-empty string.")
        return (active,)
    if not active or any(not name for name in active):
        raise ValueError("Adapter names must be non-empty strings.")
    return tuple(active)


def _entry_active_adapter(active: Any) -> ActiveAdapter | None:
    if active is None or isinstance(active, str):
        return active
    if isinstance(active, list):
        return tuple(str(name) for name in active)
    if isinstance(active, tuple):
        return active
    raise ValueError(f"Unsupported active adapter value {active!r}.")


def _init_linear(linear: eqx.nn.Linear, key: jax.Array, init: str) -> eqx.nn.Linear:
    if init == "zeros":
        weight = jnp.zeros_like(linear.weight)
    elif init == "kaiming_uniform":
        bound = jnp.sqrt(6.0 / linear.in_features)
        weight = jr.uniform(
            key,
            linear.weight.shape,
            minval=-bound,
            maxval=bound,
            dtype=linear.weight.dtype,
        )
    else:
        raise ValueError(f"Unsupported adapter init {init!r}.")
    linear = eqx.tree_at(lambda m: m.weight, linear, weight)
    if linear.bias is not None:
        linear = eqx.tree_at(lambda m: m.bias, linear, jnp.zeros_like(linear.bias))
    return linear


def _zero_linear(linear: eqx.nn.Linear) -> eqx.nn.Linear:
    linear = eqx.tree_at(lambda m: m.weight, linear, jnp.zeros_like(linear.weight))
    if linear.bias is not None:
        linear = eqx.tree_at(lambda m: m.bias, linear, jnp.zeros_like(linear.bias))
    return linear


def _apply_last_axis(module: eqx.nn.Linear, x: jax.Array) -> jax.Array:
    if x.ndim == 1:
        return module(x)
    leading_shape = x.shape[:-1]
    x_flat = x.reshape((-1, x.shape[-1]))
    y_flat = jax.vmap(module)(x_flat)
    return y_flat.reshape((*leading_shape, y_flat.shape[-1]))


def _activation(name: ActivationName):
    if name == "gelu":
        return jax.nn.gelu
    if name == "relu":
        return jax.nn.relu
    if name == "silu":
        return jax.nn.silu
    if name == "tanh":
        return jnp.tanh
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Unsupported adapter activation {name!r}.")


def _dropout(x: jax.Array, rate: float, key: jax.Array) -> jax.Array:
    keep_prob = 1.0 - rate
    mask = jr.bernoulli(key, keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0)


def _call_base(base, x, *args, key, inference, **kwargs):
    call_kwargs = dict(kwargs)
    if key is not None:
        call_kwargs["key"] = key
    if inference is not None:
        call_kwargs["inference"] = inference
    try:
        return base(x, *args, **call_kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        call_kwargs.pop("inference", None)
        return base(x, *args, **call_kwargs)


def _split_optional_key(key: jax.Array | None, count: int) -> tuple[jax.Array | None, ...]:
    if key is None:
        return (None,) * count
    return tuple(jr.split(key, count))


def _split_pair(key: jax.Array | None) -> tuple[jax.Array | None, jax.Array | None]:
    if key is None:
        return None, None
    key_a, key_b = jr.split(key, 2)
    return key_a, key_b


__all__ = (
    "AdaptFormerAdapter",
    "AdaptFormerBlock",
    "AdaptFormerConfig",
    "AdapterBankConfig",
    "AdapterConfig",
    "AdapterFusion",
    "AdapterFusionConfig",
    "AdapterRecipe",
    "BottleneckAdapter",
    "ConvPassAdapter",
    "ConvPassBlock",
    "ConvPassConfig",
    "OrthogonalLinear",
    "OutputAdapterModule",
    "OrthogonalAdapterConfig",
    "ParallelAdapterConfig",
    "ParallelAdapterBlock",
    "RepAdapterConfig",
    "RepAdapterLinear",
    "SerialAdapterBlock",
    "add_adapter",
    "adapter_fusion_trainable_spec",
    "apply_adapters",
    "apply_adaptformer",
    "apply_adapter_fusion",
    "apply_convpass",
    "apply_orthogonal_adapters",
    "apply_repadapter",
    "configure_adapter_bank",
    "extract_adapter_delta",
    "iter_adapter_wrappers",
    "iter_orthogonal_adapters",
    "iter_repadapters",
    "load_adapter_delta",
    "merge_orthogonal_adapters",
    "merge_repadapters",
    "set_active_adapter",
    "strip_adapters",
    "unmerge_orthogonal_adapters",
    "unmerge_repadapters",
)
