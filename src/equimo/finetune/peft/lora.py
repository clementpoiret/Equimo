"""LoRA modules and model surgery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..config import (
    CalibrationArtifact,
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
class RandLoRAConfig(LoRAConfig):
    """RandLoRA frozen-random-basis composition configuration."""

    rank: int = 1
    alpha: float = 1.0
    basis_count: int = 4
    seed: int | None = None
    init_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("RandLoRA rank must be >= 1.")
        if self.basis_count < 1:
            raise ValueError("RandLoRA basis_count must be >= 1.")
        if self.init_scale <= 0.0:
            raise ValueError("RandLoRA init_scale must be positive.")


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

    num_coefficients: int
    frequency_selection: Literal["random", "low_frequency", "explicit"] = "random"
    coefficient_dtype: str = "float32"
    reconstruction: Literal["conjugate_symmetric", "real_projection"] = "conjugate_symmetric"
    scale: float = 1.0
    seed: int | None = None
    frequency_indices: tuple[int, ...] = ()
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
    metadata: tuple[tuple[str, str], ...] = eqx.field(static=True)

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
        metadata: tuple[tuple[str, str], ...] = (),
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
        self.metadata = metadata

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
            raise ValueError("LoRA module is already merged.")
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
            metadata=self.metadata,
        )


class LoRAMergedLinear(LoRALinear):
    """LoRA wrapper for fused projections such as QKV linears."""


class LoRAFALinear(eqx.Module):
    """LoRA-FA wrapper with frozen A and trainable B."""

    base: eqx.Module
    frozen_A: jax.Array
    lora_fa_B: jax.Array
    correction_matrix: jax.Array
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    scaling_mode: ScalingMode = eqx.field(static=True)
    gradient_mode: Literal["frozen_A", "corrected_v3"] = eqx.field(static=True)
    gram_ridge: float = eqx.field(static=True)
    custom_vjp: bool = eqx.field(static=True)
    train_base: bool = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)
    merged: bool = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.Module,
        *,
        rank: int,
        alpha: float,
        scaling: ScalingMode,
        A_init: Literal["gaussian", "orthonormal_rows"],
        gradient_mode: Literal["frozen_A", "corrected_v3"],
        gram_ridge: float,
        custom_vjp: bool,
        train_base: bool,
        mergeable: bool,
        key: jax.Array,
        frozen_A: jax.Array | None = None,
        lora_fa_B: jax.Array | None = None,
        correction_matrix: jax.Array | None = None,
        merged: bool = False,
    ):
        if rank < 1:
            raise ValueError("LoRA-FA rank must be >= 1.")
        if gradient_mode not in {"frozen_A", "corrected_v3"}:
            raise ValueError("LoRAFAConfig.gradient_mode must be 'frozen_A' or 'corrected_v3'.")
        weight = _linear_weight(base)
        out_features, in_features = weight.shape
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling_mode = scaling
        self.gradient_mode = gradient_mode
        self.gram_ridge = gram_ridge
        self.custom_vjp = custom_vjp
        self.train_base = train_base
        self.mergeable = mergeable
        self.merged = merged
        self.frozen_A = (
            _init_lora_fa_A(rank, in_features, key, weight.dtype, A_init)
            if frozen_A is None
            else frozen_A
        )
        self.lora_fa_B = (
            jnp.zeros((out_features, rank), dtype=weight.dtype)
            if lora_fa_B is None
            else lora_fa_B
        )
        self.correction_matrix = (
            _lora_fa_correction_matrix(self.frozen_A, gram_ridge)
            if correction_matrix is None
            else correction_matrix
        )

    @property
    def lora_A(self) -> jax.Array:
        """Expose the frozen A factor for inspection without making it trainable."""

        return self.frozen_A

    @property
    def lora_B(self) -> jax.Array:
        """Expose the trainable B factor using the ordinary LoRA convention."""

        return self.lora_fa_B

    @property
    def scaling(self) -> float:
        return _scaling(self.scaling_mode, self.alpha, self.rank)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x)
        if self.merged:
            return y
        if not self.custom_vjp:
            return y + (self.lora_fa_B @ (self.frozen_A @ x)) * self.scaling
        if self.gradient_mode == "corrected_v3":
            update = _lora_fa_update_corrected(
                self.frozen_A,
                self.lora_fa_B,
                x,
                jnp.asarray(self.scaling, dtype=x.dtype),
                self.correction_matrix,
            )
        else:
            update = _lora_fa_update_frozen(
                self.frozen_A,
                self.lora_fa_B,
                x,
                jnp.asarray(self.scaling, dtype=x.dtype),
            )
        return y + update

    def delta_weight(self) -> jax.Array:
        """Return the dense LoRA-FA delta in base weight layout."""

        return (self.lora_fa_B @ self.frozen_A) * self.scaling

    def corrected_B_gradient(self, raw_grad_B: jax.Array) -> jax.Array:
        """Return the v3 projected LoRA-FA gradient for the B factor."""

        if self.gradient_mode == "frozen_A":
            return raw_grad_B
        return (raw_grad_B @ self.correction_matrix) / (self.scaling**2)

    def merge(self):
        """Return a module with the LoRA-FA delta folded into ``base.weight``."""

        if not self.mergeable:
            raise ValueError("This LoRA-FA module is not mergeable.")
        if self.merged:
            raise ValueError("LoRA-FA module is already merged.")
        base = eqx.tree_at(
            lambda m: m.weight,
            self.base,
            self.base.weight + self.delta_weight().astype(self.base.weight.dtype),
        )
        return self._replace(base=base, merged=True)

    def unmerge(self):
        """Return a module with the LoRA-FA delta removed from ``base.weight``."""

        if not self.merged:
            return self
        base = eqx.tree_at(
            lambda m: m.weight,
            self.base,
            self.base.weight - self.delta_weight().astype(self.base.weight.dtype),
        )
        return self._replace(base=base, merged=False)

    def _replace(self, *, base: eqx.Module, merged: bool):
        return LoRAFALinear(
            base,
            rank=self.rank,
            alpha=self.alpha,
            scaling=self.scaling_mode,
            A_init="orthonormal_rows",
            gradient_mode=self.gradient_mode,
            gram_ridge=self.gram_ridge,
            custom_vjp=self.custom_vjp,
            train_base=self.train_base,
            mergeable=self.mergeable,
            key=jr.PRNGKey(0),
            frozen_A=self.frozen_A,
            lora_fa_B=self.lora_fa_B,
            correction_matrix=self.correction_matrix,
            merged=merged,
        )


class RandLoRALinear(eqx.Module):
    """RandLoRA wrapper with frozen random bases and trainable composition scales."""

    base: eqx.Module
    random_A: jax.Array
    random_B: jax.Array
    basis_scales: jax.Array
    rank: int = eqx.field(static=True)
    basis_count: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    scaling_mode: ScalingMode = eqx.field(static=True)
    train_base: bool = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)
    merged: bool = eqx.field(static=True)
    seed: int | None = eqx.field(static=True)
    projection_segments: tuple[ProjectionSegment, ...] = eqx.field(static=True)
    metadata: tuple[tuple[str, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.Module,
        *,
        rank: int,
        basis_count: int,
        alpha: float,
        scaling: ScalingMode,
        init_scale: float,
        train_base: bool,
        mergeable: bool,
        key: jax.Array,
        seed: int | None = None,
        random_A: jax.Array | None = None,
        random_B: jax.Array | None = None,
        basis_scales: jax.Array | None = None,
        merged: bool = False,
        projection_segments: tuple[ProjectionSegment, ...] = (),
        metadata: tuple[tuple[str, str], ...] = (),
    ):
        if rank < 1:
            raise ValueError("RandLoRA rank must be >= 1.")
        if basis_count < 1:
            raise ValueError("RandLoRA basis_count must be >= 1.")
        if init_scale <= 0.0:
            raise ValueError("RandLoRA init_scale must be positive.")
        weight = _linear_weight(base)
        out_features, in_features = weight.shape
        if seed is not None:
            key = jr.PRNGKey(seed)
        key_a, key_b = jr.split(key, 2)
        self.base = base
        self.rank = rank
        self.basis_count = basis_count
        self.alpha = alpha
        self.scaling_mode = scaling
        self.train_base = train_base
        self.mergeable = mergeable
        self.merged = merged
        self.seed = seed
        self.projection_segments = projection_segments
        self.metadata = metadata
        self.random_A = (
            jr.normal(
                key_a,
                (basis_count, rank, in_features),
                dtype=weight.dtype,
            )
            * (init_scale / jnp.sqrt(in_features))
            if random_A is None
            else random_A
        )
        self.random_B = (
            jr.normal(
                key_b,
                (basis_count, out_features, rank),
                dtype=weight.dtype,
            )
            * (init_scale / jnp.sqrt(rank))
            if random_B is None
            else random_B
        )
        self.basis_scales = (
            jnp.zeros((basis_count,), dtype=weight.dtype)
            if basis_scales is None
            else basis_scales
        )

    @property
    def scaling(self) -> float:
        return _scaling(self.scaling_mode, self.alpha, self.rank)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x)
        if self.merged:
            return y
        if x.ndim == 1:
            return y + self.delta_weight() @ x
        return y + x @ self.delta_weight().T

    def delta_weight(self) -> jax.Array:
        """Return the dense update composed from all random bases."""

        delta = jnp.einsum(
            "b,bor,bri->oi",
            self.basis_scales,
            self.random_B,
            self.random_A,
        )
        delta = _mask_projection_segments(delta * self.scaling, self.projection_segments)
        return delta.astype(_linear_weight(self.base).dtype)

    def merge(self):
        """Return a module with the RandLoRA delta folded into ``base.weight``."""

        if not self.mergeable:
            raise ValueError("This RandLoRA module is not mergeable.")
        if self.merged:
            raise ValueError("RandLoRA module is already merged.")
        base = eqx.tree_at(
            lambda m: m.weight,
            self.base,
            self.base.weight + self.delta_weight().astype(self.base.weight.dtype),
        )
        return self._replace(base=base, merged=True)

    def unmerge(self):
        """Return a module with the RandLoRA delta removed from ``base.weight``."""

        if not self.merged:
            return self
        base = eqx.tree_at(
            lambda m: m.weight,
            self.base,
            self.base.weight - self.delta_weight().astype(self.base.weight.dtype),
        )
        return self._replace(base=base, merged=False)

    def _replace(self, *, base: eqx.Module, merged: bool):
        return RandLoRALinear(
            base,
            rank=self.rank,
            basis_count=self.basis_count,
            alpha=self.alpha,
            scaling=self.scaling_mode,
            init_scale=1.0,
            train_base=self.train_base,
            mergeable=self.mergeable,
            key=jr.PRNGKey(0),
            seed=self.seed,
            random_A=self.random_A,
            random_B=self.random_B,
            basis_scales=self.basis_scales,
            merged=merged,
            projection_segments=self.projection_segments,
            metadata=self.metadata,
        )


class FourierFTLinear(eqx.Module):
    """Sparse spectral delta wrapper for linear-like modules."""

    base: eqx.Module
    frequency_indices: jax.Array
    coefficients_real: jax.Array
    coefficients_imag: jax.Array
    weight_shape: tuple[int, int] = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    reconstruction: Literal["conjugate_symmetric", "real_projection"] = eqx.field(static=True)
    train_base: bool = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)
    merged: bool = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.Module,
        *,
        frequency_indices: jax.Array,
        coefficient_dtype: str,
        reconstruction: Literal["conjugate_symmetric", "real_projection"],
        scale: float,
        train_base: bool,
        mergeable: bool,
        coefficients_real: jax.Array | None = None,
        coefficients_imag: jax.Array | None = None,
        merged: bool = False,
    ):
        weight = _linear_weight(base)
        dtype = _coefficient_dtype(coefficient_dtype)
        self.base = base
        self.frequency_indices = frequency_indices.astype(jnp.int32)
        self.weight_shape = tuple(weight.shape)
        self.scale = scale
        self.reconstruction = reconstruction
        self.train_base = train_base
        self.mergeable = mergeable
        self.merged = merged
        self.coefficients_real = (
            jnp.zeros((self.frequency_indices.shape[0],), dtype=dtype)
            if coefficients_real is None
            else coefficients_real
        )
        self.coefficients_imag = (
            jnp.zeros((self.frequency_indices.shape[0],), dtype=dtype)
            if coefficients_imag is None
            else coefficients_imag
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x)
        if self.merged:
            return y
        return y + self.delta_weight() @ x

    def delta_weight(self) -> jax.Array:
        """Reconstruct the real dense update by inverse DFT."""

        size = self.weight_shape[0] * self.weight_shape[1]
        coeffs = self.coefficients_real + 1j * self.coefficients_imag
        spectrum = jnp.zeros((size,), dtype=coeffs.dtype)
        indices = self.frequency_indices % size
        if self.reconstruction == "conjugate_symmetric":
            self_conjugate = (indices == 0) | ((size % 2 == 0) & (indices == size // 2))
            coeffs = jnp.where(self_conjugate, self.coefficients_real.astype(coeffs.dtype), coeffs)
            spectrum = spectrum.at[indices].add(coeffs)
            conjugate_indices = (-indices) % size
            mirrored = jnp.where(self_conjugate, 0, jnp.conj(coeffs))
            spectrum = spectrum.at[conjugate_indices].add(mirrored)
        elif self.reconstruction == "real_projection":
            spectrum = spectrum.at[indices].add(coeffs)
        else:
            raise ValueError(f"Unsupported FourierFT reconstruction {self.reconstruction!r}.")
        delta = jnp.fft.ifft(spectrum, n=size).real.reshape(self.weight_shape)
        return (delta * self.scale).astype(_linear_weight(self.base).dtype)

    def merge(self):
        """Return a module with the FourierFT delta folded into ``base.weight``."""

        if not self.mergeable:
            raise ValueError("This FourierFT module is not mergeable.")
        if self.merged:
            raise ValueError("FourierFT module is already merged.")
        base = eqx.tree_at(
            lambda m: m.weight,
            self.base,
            self.base.weight + self.delta_weight().astype(self.base.weight.dtype),
        )
        return self._replace(base=base, merged=True)

    def unmerge(self):
        """Return a module with the FourierFT delta removed from ``base.weight``."""

        if not self.merged:
            return self
        base = eqx.tree_at(
            lambda m: m.weight,
            self.base,
            self.base.weight - self.delta_weight().astype(self.base.weight.dtype),
        )
        return self._replace(base=base, merged=False)

    def _replace(self, *, base: eqx.Module, merged: bool):
        return FourierFTLinear(
            base,
            frequency_indices=self.frequency_indices,
            coefficient_dtype=str(self.coefficients_real.dtype),
            reconstruction=self.reconstruction,
            scale=self.scale,
            train_base=self.train_base,
            mergeable=self.mergeable,
            coefficients_real=self.coefficients_real,
            coefficients_imag=self.coefficients_imag,
            merged=merged,
        )


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


def apply_lora_fa(
    model: PyTree,
    config: LoRAFAConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply LoRA-FA wrappers with frozen A and trainable B."""

    config = LoRAFAConfig() if config is None else config
    module_paths = _target_linear_module_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(module_paths))
    updated = model
    for module_path, subkey in zip(module_paths, keys, strict=True):
        module = get_path(updated, module_path)
        if isinstance(module, LoRAFALinear):
            continue
        if not _is_linear_like(module):
            raise TypeError(
                f"LoRA-FA target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected a linear-like module."
            )
        if config.fan_in_fan_out:
            raise ValueError("LoRA-FA currently requires canonical out_in weight layout.")
        updated = eqx.tree_at(
            lambda tree, path=module_path: get_path(tree, path),
            updated,
            LoRAFALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                scaling=config.scaling,
                A_init=config.A_init,
                gradient_mode=config.gradient_mode,
                gram_ridge=config.gram_ridge,
                custom_vjp=config.custom_vjp,
                train_base=config.train_base,
                mergeable=config.mergeable,
                key=subkey,
            ),
        )
    return updated


def apply_randlora(
    model: PyTree,
    config: RandLoRAConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply RandLoRA wrappers with frozen random low-rank bases."""

    config = RandLoRAConfig() if config is None else config
    module_paths = _target_linear_module_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(module_paths))
    updated = model
    for module_path, subkey in zip(module_paths, keys, strict=True):
        module = get_path(updated, module_path)
        if isinstance(module, RandLoRALinear):
            continue
        if not _is_linear_like(module):
            raise TypeError(
                f"RandLoRA target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected a linear-like module."
            )
        if config.fan_in_fan_out:
            raise ValueError("RandLoRA currently requires canonical out_in weight layout.")
        updated = eqx.tree_at(
            lambda tree, path=module_path: get_path(tree, path),
            updated,
            RandLoRALinear(
                module,
                rank=config.rank,
                basis_count=config.basis_count,
                alpha=config.alpha,
                scaling=config.scaling,
                init_scale=config.init_scale,
                train_base=config.train_base,
                mergeable=config.mergeable,
                key=subkey,
                seed=config.seed,
                projection_segments=_projection_segments_for_target(module, config.target),
                metadata=(
                    ("method", "randlora"),
                    ("basis_count", str(config.basis_count)),
                    ("rank", str(config.rank)),
                    ("seed", str(config.seed)),
                    ("composition", "sum_i scale_i * B_i @ A_i"),
                ),
            ),
        )
    return updated


def apply_eva_lora(
    model: PyTree,
    config: EVAInitializerConfig,
    *,
    activation_artifacts: Mapping[str, jax.Array | CalibrationArtifact],
    key: jax.Array,
    target: TargetSpec | None = None,
    lora_alpha: float = 16.0,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Initialize LoRA A factors from EVA activation right-singular directions."""

    target = LoRAConfig().target if target is None else target
    module_paths = _target_linear_module_paths(model, target, tagger=tagger)
    activation_statistics = _eva_activation_statistics(
        module_paths,
        activation_artifacts,
        config,
    )
    rank_pattern = _eva_rank_allocation(
        module_paths,
        activation_statistics,
        rank_budget=config.rank_budget,
        per_layer_min_rank=config.per_layer_min_rank,
        per_layer_max_rank=config.per_layer_max_rank,
        activation_centering=config.activation_centering,
    )
    keys = jr.split(key, len(module_paths))
    updated = model
    for module_path, subkey in zip(module_paths, keys, strict=True):
        module = get_path(updated, module_path)
        if isinstance(module, LoRALinear):
            continue
        if not _is_linear_like(module):
            raise TypeError(
                f"EVA target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected a linear-like module."
            )
        rank = rank_pattern[path_to_str(module_path)]
        if rank < 1:
            continue
        path_name = path_to_str(module_path)
        lora_A = _eva_lora_A(
            activation_statistics[path_name],
            rank,
            dtype=_linear_weight(module).dtype,
            center=config.activation_centering,
            fallback_key=subkey,
        )
        lora_B = jnp.zeros((_linear_weight(module).shape[0], rank), dtype=_linear_weight(module).dtype)
        updated = eqx.tree_at(
            lambda tree, path=module_path: get_path(tree, path),
            updated,
            LoRALinear(
                module,
                rank=rank,
                alpha=lora_alpha,
                scaling="alpha_over_r",
                dropout=0.0,
                train_base=False,
                mergeable=True,
                key=subkey,
                lora_A=lora_A,
                lora_B=lora_B,
                metadata=(("method", "eva"), *_eva_artifact_metadata(activation_artifacts[path_name])),
            ),
        )
    return updated


def apply_loftq_lora(
    model: PyTree,
    config: LoftQConfig,
    *,
    quantized_weights: Mapping[str, jax.Array],
    key: jax.Array,
    target: TargetSpec | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply LoftQ initialization around externally quantized base weights."""

    del key
    target = LoRAConfig().target if target is None else target
    module_paths = _target_linear_module_paths(model, target, tagger=tagger)
    updated = model
    for module_path in module_paths:
        path_name = path_to_str(module_path)
        module = get_path(updated, module_path)
        if isinstance(module, LoRALinear):
            continue
        if not _is_linear_like(module):
            raise TypeError(
                f"LoftQ target {path_name!r} is {type(module).__name__}, "
                "expected a linear-like module."
            )
        weight = _linear_weight(module)
        q_weight = quantized_weights.get(path_name)
        if q_weight is None:
            raise ValueError(f"Missing quantized weight for LoftQ target {path_name!r}.")
        if tuple(q_weight.shape) != tuple(weight.shape):
            raise ValueError(
                f"LoftQ quantized weight for {path_name!r} has shape "
                f"{tuple(q_weight.shape)}, expected {tuple(weight.shape)}."
            )
        fingerprint = loftq_quantization_fingerprint(path_name, q_weight, config)
        residual = (weight - q_weight.astype(weight.dtype)).astype(jnp.float32)
        lora_A, lora_B = _svd_lora_from_residual(
            residual,
            rank=config.rank,
            scaling=config.scaling,
            dtype=weight.dtype,
        )
        quantized_base = eqx.tree_at(
            lambda linear: linear.weight,
            module,
            q_weight.astype(weight.dtype),
        )
        updated = eqx.tree_at(
            lambda tree, path=module_path: get_path(tree, path),
            updated,
            LoRALinear(
                quantized_base,
                rank=int(lora_A.shape[0]),
                alpha=float(config.scaling * lora_A.shape[0]),
                scaling="alpha_over_r",
                dropout=0.0,
                train_base=False,
                mergeable=True,
                key=jr.PRNGKey(0),
                lora_A=lora_A,
                lora_B=lora_B,
                metadata=(
                    ("method", "loftq"),
                    ("quantizer_id", config.quantizer.id),
                    ("quantizer_bits", str(config.quantizer.bits)),
                    ("quantizer_format", config.quantizer.format),
                    ("iterations", str(config.iterations)),
                    ("quantization_fingerprint", fingerprint),
                ),
            ),
        )
    return updated


def validate_loftq_lora(
    model: PyTree,
    config: LoftQConfig,
    *,
    quantized_weights: Mapping[str, jax.Array],
    target: TargetSpec | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> None:
    """Validate that LoftQ adapters match the supplied quantized base payload."""

    target = LoRAConfig().target if target is None else target
    module_paths = _target_linear_module_paths(model, target, tagger=tagger)
    seen: set[str] = set()
    for module_path in module_paths:
        module_path, module = _nearest_lora_owner(model, module_path)
        path_name = path_to_str(module_path)
        if path_name in seen:
            continue
        seen.add(path_name)
        if not isinstance(module, LoRALinear):
            raise ValueError(f"LoftQ target {path_name!r} is not a LoRA module.")
        q_weight = quantized_weights.get(path_name)
        if q_weight is None:
            raise ValueError(f"Missing quantized weight for LoftQ target {path_name!r}.")
        expected = loftq_quantization_fingerprint(path_name, q_weight, config)
        metadata = dict(module.metadata)
        actual = metadata.get("quantization_fingerprint")
        if actual != expected:
            raise ValueError(
                f"LoftQ quantization fingerprint mismatch for {path_name!r}: "
                f"expected {expected}, got {actual}."
            )


def _nearest_lora_owner(model: PyTree, path: Path) -> tuple[Path, Any]:
    for end in range(len(path), 0, -1):
        candidate_path = path[:end]
        candidate = get_path(model, candidate_path)
        if isinstance(candidate, LoRALinear):
            return candidate_path, candidate
    return path, get_path(model, path)


def loftq_quantization_fingerprint(
    logical_id: str,
    quantized_weight: jax.Array,
    config: LoftQConfig,
) -> str:
    """Return the lineage fingerprint for a LoftQ quantized base payload."""

    array = jax.device_get(jnp.asarray(quantized_weight))
    digest = hashlib.sha256()
    digest.update(str(logical_id).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    digest.update(config.quantizer.id.encode())
    digest.update(str(config.quantizer.bits).encode())
    digest.update(config.quantizer.format.encode())
    digest.update(str(config.quantizer.compute_dtype).encode())
    digest.update(str(config.iterations).encode())
    digest.update(str(config.scaling).encode())
    digest.update(config.residual_svd.encode())
    return digest.hexdigest()


def apply_fourierft(
    model: PyTree,
    config: FourierFTConfig,
    *,
    key: jax.Array | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply FourierFT sparse spectral wrappers to selected linears."""

    module_paths = _target_linear_module_paths(model, config.target, tagger=tagger)
    if key is None:
        key = jr.PRNGKey(0 if config.seed is None else config.seed)
    keys = jr.split(key, len(module_paths))
    updated = model
    for module_path, subkey in zip(module_paths, keys, strict=True):
        module = get_path(updated, module_path)
        if isinstance(module, FourierFTLinear):
            continue
        if not _is_linear_like(module):
            raise TypeError(
                f"FourierFT target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected a linear-like module."
            )
        weight = _linear_weight(module)
        indices = _fourier_frequency_indices(
            weight.size,
            config.num_coefficients,
            config.frequency_selection,
            explicit=config.frequency_indices,
            key=subkey,
        )
        updated = eqx.tree_at(
            lambda tree, path=module_path: get_path(tree, path),
            updated,
            FourierFTLinear(
                module,
                frequency_indices=indices,
                coefficient_dtype=config.coefficient_dtype,
                reconstruction=config.reconstruction,
                scale=config.scale,
                train_base=False,
                mergeable=True,
            ),
        )
    return updated


def merge_fourierft(model: PyTree) -> PyTree:
    """Merge every FourierFT module in ``model``."""

    return _map_fourierft_modules(model, lambda module: module.merge())


def unmerge_fourierft(model: PyTree) -> PyTree:
    """Unmerge every merged FourierFT module in ``model``."""

    return _map_fourierft_modules(model, lambda module: module.unmerge())


def merge_lora(model: PyTree) -> PyTree:
    """Merge every LoRA module in ``model``."""

    return _map_lora_modules(model, lambda module: module.merge())


def merge_lora_fa(model: PyTree) -> PyTree:
    """Merge every LoRA-FA module in ``model``."""

    return _map_lora_fa_modules(model, lambda module: module.merge())


def merge_randlora(model: PyTree) -> PyTree:
    """Merge every RandLoRA module in ``model``."""

    return _map_randlora_modules(model, lambda module: module.merge())


def unmerge_lora_fa(model: PyTree) -> PyTree:
    """Unmerge every merged LoRA-FA module in ``model``."""

    return _map_lora_fa_modules(model, lambda module: module.unmerge())


def unmerge_randlora(model: PyTree) -> PyTree:
    """Unmerge every merged RandLoRA module in ``model``."""

    return _map_randlora_modules(model, lambda module: module.unmerge())


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
        metadata = dict(module.metadata)
        base_weight_delta = module.base_weight_delta
        if metadata.get("method") == "loftq" and base_weight_delta is None:
            base_weight_delta = -module.delta_weight().astype(_linear_weight(module.base).dtype)
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
                "base_weight_delta": base_weight_delta,
                "metadata": module.metadata,
            }
        )
    for path, module in iter_randlora_modules(model):
        entries.append(
            {
                "path": path_to_str(path),
                "class": "RandLoRALinear",
                "rank": module.rank,
                "basis_count": module.basis_count,
                "alpha": module.alpha,
                "scaling": module.scaling_mode,
                "train_base": module.train_base,
                "mergeable": module.mergeable,
                "factor_convention": "delta = sum_i scale_i * B_i @ A_i",
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
                "seed": module.seed,
                "base_weight_shape": tuple(_linear_weight(module.base).shape),
                "base_bias_shape": None
                if _linear_bias(module.base) is None
                else tuple(_linear_bias(module.base).shape),
                "random_A": module.random_A,
                "random_B": module.random_B,
                "basis_scales": module.basis_scales,
                "metadata": module.metadata,
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
        if entry["class"] == "RandLoRALinear":
            entry_metadata = _entry_metadata(entry)
            lora_module = RandLoRALinear(
                module,
                rank=int(entry["rank"]),
                basis_count=int(entry["basis_count"]),
                alpha=float(entry["alpha"]),
                scaling=entry["scaling"],
                init_scale=1.0,
                train_base=bool(entry["train_base"]),
                mergeable=bool(entry["mergeable"]),
                key=jr.PRNGKey(0),
                seed=entry.get("seed"),
                random_A=entry["random_A"],
                random_B=entry["random_B"],
                basis_scales=entry["basis_scales"],
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
                metadata=tuple(sorted(entry_metadata.items())),
            )
            if entry["merged"]:
                lora_module = lora_module.merge()
            updated = eqx.tree_at(
                lambda tree, p=path: get_path(tree, p),
                updated,
                lora_module,
            )
            continue
        base_weight_delta = entry.get("base_weight_delta")
        if base_weight_delta is not None:
            module = eqx.tree_at(
                lambda linear: linear.weight,
                module,
                _linear_weight(module) + base_weight_delta.astype(_linear_weight(module).dtype),
            )

        entry_metadata = _entry_metadata(entry)
        _validate_quantized_entry_metadata(entry["path"], entry_metadata)
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
            metadata=tuple(sorted(entry_metadata.items())),
        )
        if entry["merged"]:
            lora_module = lora_module.merge()
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, lora_module)

    return updated


def strip_lora(model: PyTree) -> PyTree:
    """Replace LoRA wrappers with their unmerged base linears."""

    stripped = unmerge_lora(model)
    stripped = unmerge_randlora(stripped)
    for path, module in iter_lora_modules(stripped):
        base = _restore_base_weight(module)
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, base)
    for path, module in iter_randlora_modules(stripped):
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


def iter_lora_fa_modules(model: PyTree) -> tuple[tuple[Path, LoRAFALinear], ...]:
    """Return path/module pairs for LoRA-FA wrappers in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, LoRAFALinear),
        )
        if isinstance(leaf, LoRAFALinear)
    )


def iter_fourierft_modules(model: PyTree) -> tuple[tuple[Path, FourierFTLinear], ...]:
    """Return path/module pairs for FourierFT wrappers in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, FourierFTLinear),
        )
        if isinstance(leaf, FourierFTLinear)
    )


def iter_randlora_modules(model: PyTree) -> tuple[tuple[Path, RandLoRALinear], ...]:
    """Return path/module pairs for RandLoRA wrappers in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, RandLoRALinear),
        )
        if isinstance(leaf, RandLoRALinear)
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


def _entry_metadata(entry: Mapping[str, Any]) -> dict[str, str]:
    metadata = entry.get("metadata", ())
    if isinstance(metadata, Mapping):
        return {str(key): str(value) for key, value in metadata.items()}
    return {str(key): str(value) for key, value in tuple(metadata)}


def _validate_quantized_entry_metadata(path: str, metadata: Mapping[str, str]) -> None:
    if metadata.get("method") != "loftq":
        return
    if not metadata.get("quantization_fingerprint"):
        raise FineTuneBundleError(
            f"LoftQ delta entry {path!r} is missing quantization_fingerprint."
        )


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


def _init_lora_fa_A(
    rank: int,
    in_features: int,
    key: jax.Array,
    dtype,
    init: Literal["gaussian", "orthonormal_rows"],
) -> jax.Array:
    if init == "gaussian":
        return (jr.normal(key, (rank, in_features), dtype=dtype) / jnp.sqrt(in_features)).astype(dtype)
    if init != "orthonormal_rows":
        raise ValueError("LoRAFAConfig.A_init must be 'gaussian' or 'orthonormal_rows'.")
    matrix = jr.normal(key, (in_features, rank), dtype=jnp.float32)
    q, _ = jnp.linalg.qr(matrix, mode="reduced")
    return q.T.astype(dtype)


@jax.custom_vjp
def _lora_fa_update_frozen(
    A: jax.Array,
    B: jax.Array,
    x: jax.Array,
    scaling: jax.Array,
) -> jax.Array:
    return (B @ (A @ x)) * scaling


def _lora_fa_update_frozen_fwd(A, B, x, scaling):
    z = A @ x
    return (B @ z) * scaling, (A, B, z, scaling)


def _lora_fa_update_frozen_bwd(residual, cotangent):
    A, B, z, scaling = residual
    grad_A = jnp.zeros_like(A)
    grad_B = jnp.outer(cotangent, z) * scaling
    grad_x = A.T @ (B.T @ cotangent) * scaling
    return grad_A, grad_B, grad_x, None


_lora_fa_update_frozen.defvjp(
    _lora_fa_update_frozen_fwd,
    _lora_fa_update_frozen_bwd,
)


@jax.custom_vjp
def _lora_fa_update_corrected(
    A: jax.Array,
    B: jax.Array,
    x: jax.Array,
    scaling: jax.Array,
    correction_matrix: jax.Array,
) -> jax.Array:
    return (B @ (A @ x)) * scaling


def _lora_fa_update_corrected_fwd(A, B, x, scaling, correction_matrix):
    z = A @ x
    return (B @ z) * scaling, (A, B, z, scaling, correction_matrix)


def _lora_fa_update_corrected_bwd(residual, cotangent):
    A, B, z, scaling, correction_matrix = residual
    raw_grad_B = jnp.outer(cotangent, z) * scaling
    grad_A = jnp.zeros_like(A)
    grad_B = (raw_grad_B @ correction_matrix) / (scaling**2)
    grad_x = A.T @ (B.T @ cotangent) * scaling
    return grad_A, grad_B, grad_x, None, None


_lora_fa_update_corrected.defvjp(
    _lora_fa_update_corrected_fwd,
    _lora_fa_update_corrected_bwd,
)


def _lora_fa_correction_matrix(A: jax.Array, gram_ridge: float) -> jax.Array:
    A32 = A.astype(jnp.float32)
    gram = A32 @ A32.T
    eye = jnp.eye(gram.shape[0], dtype=jnp.float32)
    return jnp.linalg.solve(gram + gram_ridge * eye, eye).astype(A.dtype)


def _coefficient_dtype(name: str):
    if name in {"float32", "f32"}:
        return jnp.float32
    if name in {"float64", "f64"}:
        return jnp.float64
    if name in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    raise ValueError(f"Unsupported FourierFT coefficient dtype {name!r}.")


def _fourier_frequency_indices(
    size: int,
    count: int,
    selection: Literal["random", "low_frequency", "explicit"],
    *,
    explicit: tuple[int, ...],
    key: jax.Array,
) -> jax.Array:
    if size < 1:
        raise ValueError("FourierFT requires a non-empty weight.")
    if count < 1:
        raise ValueError("FourierFT num_coefficients must be >= 1.")
    if selection == "explicit":
        if len(explicit) != count:
            raise ValueError("Explicit FourierFT frequencies must match num_coefficients.")
        indices = jnp.asarray(explicit, dtype=jnp.int32) % size
    elif selection == "low_frequency":
        indices = jnp.arange(min(count, size), dtype=jnp.int32)
    elif selection == "random":
        indices = jr.choice(key, size, shape=(min(count, size),), replace=False).astype(jnp.int32)
    else:
        raise ValueError(f"Unsupported FourierFT frequency_selection {selection!r}.")
    canonical = jnp.minimum(indices, (-indices) % size)
    _, unique_positions = jnp.unique(canonical, size=canonical.shape[0], return_index=True)
    unique_positions = jnp.sort(unique_positions)
    return indices[unique_positions]


def _eva_activation_statistics(
    module_paths: tuple[Path, ...],
    activation_artifacts: Mapping[str, jax.Array | CalibrationArtifact],
    config: EVAInitializerConfig,
) -> dict[str, jax.Array]:
    names = tuple(path_to_str(path) for path in module_paths)
    missing = sorted(set(names) - set(activation_artifacts))
    if missing:
        raise ValueError(f"Missing EVA activation artifacts for: {', '.join(missing)}.")
    return {
        name: _eva_statistics_array(name, activation_artifacts[name], config)
        for name in names
    }


def _eva_statistics_array(
    logical_id: str,
    artifact: jax.Array | CalibrationArtifact,
    config: EVAInitializerConfig,
) -> jax.Array:
    if isinstance(artifact, CalibrationArtifact):
        _validate_eva_calibration_artifact(logical_id, artifact, config)
        return _eva_statistics_payload_array(logical_id, artifact.statistics)
    if config.calibration is not None:
        raise ValueError(
            "EVA calibration requires immutable CalibrationArtifact entries; "
            f"got bare statistics for {logical_id!r}."
        )
    return jnp.asarray(artifact)


def _eva_statistics_payload_array(logical_id: str, statistics: PyTree) -> jax.Array:
    if isinstance(statistics, Mapping):
        if "activations" in statistics:
            return jnp.asarray(statistics["activations"])
        if "input_activations" in statistics:
            return jnp.asarray(statistics["input_activations"])
        if "right_singular_vectors" in statistics:
            vh = jnp.asarray(statistics["right_singular_vectors"])
            singular_values = statistics.get("singular_values")
            return _svd_statistics_matrix(vh, singular_values)
        if "vh" in statistics:
            vh = jnp.asarray(statistics["vh"])
            singular_values = statistics.get("singular_values")
            return _svd_statistics_matrix(vh, singular_values)
        raise ValueError(
            f"EVA calibration statistics for {logical_id!r} must include "
            "'activations', 'input_activations', 'right_singular_vectors', or 'vh'."
        )
    return jnp.asarray(statistics)


def _svd_statistics_matrix(
    right_singular_vectors: jax.Array,
    singular_values: Any | None,
) -> jax.Array:
    if singular_values is None:
        return right_singular_vectors
    singular_values = jnp.asarray(singular_values, dtype=jnp.float32)
    count = min(int(right_singular_vectors.shape[0]), int(singular_values.shape[0]))
    return singular_values[:count, None] * right_singular_vectors[:count].astype(jnp.float32)


def _validate_eva_calibration_artifact(
    logical_id: str,
    artifact: CalibrationArtifact,
    config: EVAInitializerConfig,
) -> None:
    allowed_kinds = {"activation_svd", "activation_covariance"}
    if artifact.kind not in allowed_kinds:
        raise ValueError(
            f"EVA calibration artifact {logical_id!r} has kind {artifact.kind!r}; "
            f"expected one of {tuple(sorted(allowed_kinds))!r}."
        )
    if not artifact.base_checkpoint_hash:
        raise ValueError(f"EVA calibration artifact {logical_id!r} is missing base_checkpoint_hash.")
    if artifact.logical_parameter_ids and logical_id not in artifact.logical_parameter_ids:
        raise ValueError(
            f"EVA calibration artifact for {logical_id!r} does not include that logical ID."
        )
    if artifact.sample_count <= 0:
        raise ValueError(f"EVA calibration artifact {logical_id!r} must have sample_count > 0.")
    if artifact.accumulation_dtype not in {"float32", "fp32", "float64", "fp64"}:
        raise ValueError(
            f"EVA calibration artifact {logical_id!r} must accumulate statistics in fp32 or fp64."
        )
    if not artifact.data_fingerprint:
        raise ValueError(f"EVA calibration artifact {logical_id!r} is missing data_fingerprint.")
    if not artifact.distributed_reduction:
        raise ValueError(f"EVA calibration artifact {logical_id!r} is missing distributed_reduction.")
    if config.calibration is None:
        return
    if artifact.kind != config.calibration.artifact_kind:
        raise ValueError(
            f"EVA calibration artifact {logical_id!r} kind mismatch: "
            f"expected {config.calibration.artifact_kind!r}, got {artifact.kind!r}."
        )
    if (
        config.calibration.sample_count is not None
        and artifact.sample_count != config.calibration.sample_count
    ):
        raise ValueError(
            f"EVA calibration artifact {logical_id!r} sample_count mismatch: "
            f"expected {config.calibration.sample_count}, got {artifact.sample_count}."
        )
    if (
        config.calibration.data_fingerprint is not None
        and artifact.data_fingerprint != config.calibration.data_fingerprint
    ):
        raise ValueError(
            f"EVA calibration artifact {logical_id!r} data_fingerprint mismatch: "
            f"expected {config.calibration.data_fingerprint!r}, "
            f"got {artifact.data_fingerprint!r}."
        )


def _eva_artifact_metadata(
    artifact: jax.Array | CalibrationArtifact,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(artifact, CalibrationArtifact):
        return ()
    return (
        ("calibration_kind", artifact.kind),
        ("calibration_sample_count", str(artifact.sample_count)),
        ("calibration_data_fingerprint", artifact.data_fingerprint),
        ("calibration_reduction", artifact.distributed_reduction),
        ("calibration_base_checkpoint_hash", artifact.base_checkpoint_hash),
    )


def _eva_rank_allocation(
    module_paths: tuple[Path, ...],
    activation_artifacts: Mapping[str, jax.Array],
    *,
    rank_budget: int,
    per_layer_min_rank: int,
    per_layer_max_rank: int | None,
    activation_centering: bool,
) -> dict[str, int]:
    if rank_budget < 1:
        raise ValueError("EVA rank_budget must be >= 1.")
    if per_layer_min_rank < 0:
        raise ValueError("EVA per_layer_min_rank must be non-negative.")
    names = tuple(path_to_str(path) for path in module_paths)
    missing = sorted(set(names) - set(activation_artifacts))
    if missing:
        raise ValueError(f"Missing EVA activation artifacts for: {', '.join(missing)}.")
    max_rank = rank_budget if per_layer_max_rank is None else per_layer_max_rank
    if max_rank < 1:
        raise ValueError("EVA per_layer_max_rank must be >= 1 when provided.")
    allocation = {name: min(per_layer_min_rank, max_rank) for name in names}
    remaining = rank_budget - sum(allocation.values())
    if remaining < 0:
        raise ValueError("EVA rank_budget is smaller than the requested per-layer minimum.")
    scores: list[tuple[float, str, int]] = []
    for name in names:
        activations = _activation_matrix(activation_artifacts[name], center=activation_centering)
        _, s, _ = jnp.linalg.svd(activations.astype(jnp.float32), full_matrices=False)
        width = int(s.shape[0])
        for index in range(allocation[name], min(width, max_rank)):
            scores.append((float(s[index] ** 2), name, index))
    scores.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, name, _ in scores[:remaining]:
        allocation[name] += 1
    return allocation


def _activation_matrix(activations: jax.Array, *, center: bool) -> jax.Array:
    matrix = jnp.asarray(activations)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    matrix = matrix.reshape((-1, matrix.shape[-1]))
    matrix = matrix.astype(jnp.float32)
    if center:
        matrix = matrix - jnp.mean(matrix, axis=0, keepdims=True)
    return matrix


def _eva_lora_A(
    activations: jax.Array,
    rank: int,
    *,
    dtype,
    center: bool,
    fallback_key: jax.Array,
) -> jax.Array:
    matrix = _activation_matrix(activations, center=center)
    try:
        _, _, vh = jnp.linalg.svd(matrix, full_matrices=False)
        return vh[:rank].astype(dtype)
    except Exception:
        return _init_lora_fa_A(rank, int(matrix.shape[-1]), fallback_key, dtype, "orthonormal_rows")


def _svd_lora_from_residual(
    residual: jax.Array,
    *,
    rank: int,
    scaling: float,
    dtype,
) -> tuple[jax.Array, jax.Array]:
    if rank < 1:
        raise ValueError("LoftQ rank must be >= 1.")
    if scaling == 0.0:
        raise ValueError("LoftQ scaling must be non-zero.")
    u, s, vh = jnp.linalg.svd(residual.astype(jnp.float32), full_matrices=False)
    rank = min(rank, int(s.shape[0]))
    sqrt_s = jnp.sqrt(s[:rank] / scaling).astype(dtype)
    lora_B = u[:, :rank].astype(dtype) * sqrt_s[None, :]
    lora_A = sqrt_s[:, None] * vh[:rank].astype(dtype)
    return lora_A, lora_B


def _map_lora_modules(model: PyTree, fn) -> PyTree:
    updated = model
    for path, module in iter_lora_modules(updated):
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, fn(module))
    return updated


def _map_lora_fa_modules(model: PyTree, fn) -> PyTree:
    updated = model
    for path, module in iter_lora_fa_modules(updated):
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, fn(module))
    return updated


def _map_fourierft_modules(model: PyTree, fn) -> PyTree:
    updated = model
    for path, module in iter_fourierft_modules(updated):
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, fn(module))
    return updated


def _map_randlora_modules(model: PyTree, fn) -> PyTree:
    updated = model
    for path, module in iter_randlora_modules(updated):
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
    "FourierFTLinear",
    "LoftQConfig",
    "LoRAFAConfig",
    "LoRAFALinear",
    "QuantizerSpec",
    "RandLoRAConfig",
    "RandLoRALinear",
    "apply_adalora",
    "apply_eva_lora",
    "apply_fourierft",
    "apply_loftq_lora",
    "apply_lora_fa",
    "apply_lora_rank_pattern",
    "apply_randlora",
    "apply_lora",
    "architecture_hash",
    "extract_lora_delta",
    "iter_fourierft_modules",
    "iter_lora_fa_modules",
    "iter_lora_modules",
    "iter_randlora_modules",
    "load_lora_delta",
    "loftq_quantization_fingerprint",
    "lora_config_to_dict",
    "lora_rank_groups",
    "merge_fourierft",
    "merge_lora",
    "merge_lora_fa",
    "merge_randlora",
    "strip_lora",
    "unmerge_fourierft",
    "unmerge_lora",
    "unmerge_lora_fa",
    "unmerge_randlora",
    "validate_loftq_lora",
)
