# ty: ignore[call-non-callable]
# ty: ignore[invalid-assignment]
# ty: ignore[invalid-return-type]
# ty: ignore[too-many-positional-arguments]
# ty: ignore[unknown-argument]
# ty: ignore[unresolved-attribute]
# ty: ignore[invalid-argument-type]
"""Vision Parcae for Equimo.

A ViT-style image model with a Parcae middle loop.
    h[t + 1] = A h[t] + B e + R(h[t], e)

where ``e`` is the prelude output and ``R`` is a shared transformer block
chunk.  The default injection is the original Parcae diagonal discretisation,
which keeps the recurrent linear dynamics contractive by construction.  Use
``injection_type="diagonal_exact_zoh"`` to select the exact-ZOH variant while
leaving ``injection_type="diagonal"`` available for controlled comparisons.

This module is intentionally self-contained.  Drop it into ``equimo/models``
and expose the desired names from the package ``__init__`` if needed.
"""

__all__ = [
    "BInitMode",
    "VisionParcae",
    "VisionParcaeDiagonalInjection",
    "VisionParcaeDiagonalExactZOHInjection",
    "VisionParcaeLinearInjection",
    "VisionParcaeAdditiveInjection",
    "dynamics_from_alpha",
    "vision_parcae_tiny_patch16_224",
    "vision_parcae_small_patch16_224",
    "vision_parcae_base_patch16_224",
]

import math
from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.core.layers.activation import get_act
from equimo.vision.layers.attention import get_attn, get_attn_block
from equimo.core.layers.ffn import get_ffn
from equimo.core.layers.generic import BlockChunk
from equimo.core.layers.norm import get_norm
from equimo.vision.layers.patch import PatchEmbedding
from equimo.vision.layers.posemb import CompositeVisionRoPE, LearnedPosEmbed, VisionRoPE
from equimo.registry import register_model
from equimo.utils import pool_sd

InjectionKind = Literal["diagonal", "diagonal_exact_zoh", "linear", "add"]
BInitMode = Literal["raw", "fixed_point", "target_depth", "one_step"]
StateInitKind = Literal["like-init", "normal", "embed", "zero", "unit"]
CInitKind = Literal["scaled", "identity"]
SamplingKind = Literal[
    "fixed",
    "poisson-truncated-full",
    "poisson-full",
    "poisson-fill",
]


def _inverse_softplus(x: float | Array) -> Array:
    """Numerically stable inverse softplus for positive initialisation."""

    x = jnp.asarray(x)
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(x)))


def _exact_zoh_input_gain(dt: Array, A: Array) -> Array:
    """Return the exact zero-order-hold gain for held inputs.

    ``A`` contains positive diagonal rates for the continuous generator ``-A``.
    Using ``expm1`` computes ``(1 - exp(-dt * A)) / A`` without subtractive
    cancellation when ``dt * A`` is small.
    """

    return -jnp.expm1(-dt * A) / A


def dynamics_from_alpha(
    *,
    alpha: float,
    dt: float,
    injection_type: str,
    mode: BInitMode,
    target_depth: int | None = None,
    target_scale: float = 1.0,
    raw_B_init_scale: float | None = None,
) -> tuple[float, float]:
    """Return ``(A_init, B_init_scale)`` for alpha-guided diagonal dynamics.

    ``fixed_point`` targets the linear fixed point ``h_inf ~= target_scale * e``.
    ``target_depth`` targets ``h[target_depth] ~= target_scale * e`` from
    ``h0 = 0``.  ``one_step`` is the ``target_depth=1`` case.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    A_init = -math.log(alpha) / dt

    if mode == "raw":
        if raw_B_init_scale is None:
            raise ValueError("raw mode requires raw_B_init_scale.")
        return A_init, raw_B_init_scale

    if injection_type not in ("diagonal", "diagonal_exact_zoh"):
        raise ValueError("B_init_mode only applies to diagonal injections.")

    if mode == "one_step":
        target_depth = 1

    if mode in ("target_depth", "one_step"):
        if target_depth is None or target_depth < 1:
            raise ValueError("target_depth mode requires target_depth >= 1.")
        denom = 1.0 - alpha**target_depth
        if injection_type == "diagonal":
            B_init_scale = target_scale * (1.0 - alpha) / (dt * denom)
        else:
            B_init_scale = target_scale * A_init / denom
        return A_init, B_init_scale

    if mode == "fixed_point":
        if injection_type == "diagonal":
            B_init_scale = target_scale * (1.0 - alpha) / dt
        else:
            B_init_scale = target_scale * A_init
        return A_init, B_init_scale

    raise ValueError(f"Unknown B init mode: {mode!r}.")


def _takase_std(dim: int) -> float:
    """Takase/Karpathy scaled-normal std, ``sqrt(2 / (5 d))``."""

    if dim <= 0:
        raise ValueError("dim must be positive.")
    return math.sqrt(2.0 / (5.0 * float(dim)))


def _depth_scaled_std(fan_dim: int, depth: int = 1) -> float:
    """Depth-aware local approximation to Parcae's init manager."""

    return _takase_std(fan_dim) / math.sqrt(float(max(depth, 1)))


def _default_max_recurrence(mean_recurrence: int, sample_recurrence: bool) -> int:
    """Return a static scan bound for JIT-compatible recurrence.

    The recurrent loop is implemented as a fixed-length ``lax.scan`` plus
    dynamic gates, so XLA needs a static upper bound.  The default leaves room
    for ordinary test-time scaling (``2 * mean_recurrence``) and, when
    stochastic recurrence is enabled, a four-sigma Poisson cap.
    """

    poisson_cap = (
        int(math.ceil(mean_recurrence + 4.0 * math.sqrt(float(mean_recurrence))))
        if sample_recurrence
        else 0
    )
    return max(2 * mean_recurrence, poisson_cap, 1)


def _to_scalar_i32(x: int | Array, *, name: str) -> Array:
    """Convert a Python scalar or JAX scalar to a non-negative int32 scalar."""

    if isinstance(x, (int, np.integer)) and int(x) < 0:
        raise ValueError(f"{name} must be non-negative.")
    arr = jnp.asarray(x, dtype=jnp.int32)
    if arr.shape != ():
        raise ValueError(f"{name} must be a scalar; got shape {arr.shape}.")
    return jnp.maximum(arr, jnp.asarray(0, dtype=jnp.int32))


def _make_scaled_linear(
    in_features: int,
    out_features: int,
    *,
    use_bias: bool,
    key: PRNGKeyArray,
    std: float,
) -> eqx.nn.Linear:
    """Create ``eqx.nn.Linear`` with truncated-normal weights and zero bias."""

    key_linear, key_weight = jr.split(key, 2)
    linear = eqx.nn.Linear(
        in_features,
        out_features,
        use_bias=use_bias,
        key=key_linear,
    )
    weight = jr.truncated_normal(
        key_weight,
        lower=-3.0,
        upper=3.0,
        shape=linear.weight.shape,
    )
    weight = (weight * std).astype(linear.weight.dtype)
    linear = eqx.tree_at(lambda m: m.weight, linear, weight)
    if linear.bias is not None:
        linear = eqx.tree_at(lambda m: m.bias, linear, jnp.zeros_like(linear.bias))
    return linear


def _make_identity_or_padded_linear(
    in_features: int,
    out_features: int,
    *,
    use_bias: bool,
    key: PRNGKeyArray,
) -> eqx.nn.Linear:
    """Create an identity-padded ``eqx.nn.Linear`` with optional zero bias."""

    linear = eqx.nn.Linear(in_features, out_features, use_bias=use_bias, key=key)
    weight = jnp.zeros_like(linear.weight)
    diag = min(in_features, out_features)
    weight = weight.at[jnp.arange(diag), jnp.arange(diag)].set(
        jnp.asarray(1.0, dtype=weight.dtype)
    )
    linear = eqx.tree_at(lambda m: m.weight, linear, weight)
    if linear.bias is not None:
        linear = eqx.tree_at(lambda m: m.bias, linear, jnp.zeros_like(linear.bias))
    return linear


def _apply_linear_sequence(
    linear: eqx.nn.Linear,
    x: Float[Array, "seq in_dim"],
) -> Float[Array, "seq out_dim"]:
    return jax.vmap(linear)(x)


def _apply_norm_sequence(
    norm: eqx.Module,
    x: Float[Array, "seq dim"],
) -> Float[Array, "seq dim"]:
    return jax.vmap(norm)(x)


def _make_block_chunk(
    *,
    depth: int,
    dim: int,
    num_heads: int,
    block: type[eqx.Module],
    attn_layer: type[eqx.Module],
    ffn_layer: type[eqx.Module],
    mlp_ratio: float,
    qkv_bias: bool,
    proj_bias: bool,
    qk_norm: bool,
    attn_drop: float,
    proj_drop: float,
    act_layer: Callable,
    ffn_bias: bool,
    ffn_kwargs: dict,
    norm_layer: type[eqx.Module],
    eps: float,
    drop_path: list[float],
    init_values: float | None,
    key: PRNGKeyArray,
) -> BlockChunk | None:
    """Build a ViT-style Equimo ``BlockChunk`` or return ``None`` for depth 0."""

    if depth <= 0:
        return None

    return BlockChunk(
        depth=depth,
        module=block,
        module_kwargs={
            "dim": dim,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "qkv_bias": qkv_bias,
            "proj_bias": proj_bias,
            "qk_norm": qk_norm,
            "attn_drop": attn_drop,
            "proj_drop": proj_drop,
            "act_layer": act_layer,
            "attn_layer": attn_layer,
            "ffn_layer": ffn_layer,
            "ffn_bias": ffn_bias,
            "ffn_kwargs": ffn_kwargs,
            "norm_layer": norm_layer,
            "eps": eps,
        },
        drop_path=drop_path,
        init_values=init_values,
        key=key,
    )


class VisionParcaeDiagonalInjection(eqx.Module):
    """Parcae diagonal injection with the original Euler input write.

        x[t + 1] = exp(-softplus(dt) * exp(A_log)) * x[t]
                 + softplus(dt) * B e

    with ``B`` identity-padded at initialisation.  The recurrent state dimension
    may differ from the prelude embedding dimension.
    """

    A_log: jax.Array
    dt_bias: jax.Array
    B: jax.Array

    input_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        *,
        dt_init: float = 0.1,
        A_init: float = 1.0,
        B_init_scale: float = 1.0,
        key: PRNGKeyArray | None = None,
    ):
        if dt_init <= 0:
            raise ValueError("dt_init must be positive.")
        if A_init <= 0:
            raise ValueError("A_init must be positive.")

        del key  # B is deterministic identity-padding in the reference.
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.A_log = jnp.full((state_dim,), jnp.log(A_init), dtype=jnp.float32)
        self.dt_bias = jnp.full(
            (state_dim,),
            _inverse_softplus(dt_init),
            dtype=jnp.float32,
        )

        B = jnp.zeros((state_dim, input_dim), dtype=jnp.float32)
        diag = min(input_dim, state_dim)
        B = B.at[jnp.arange(diag), jnp.arange(diag)].set(B_init_scale)
        self.B = B

    def __call__(
        self,
        x_t: Float[Array, "seq state_dim"],
        e: Float[Array, "seq input_dim"],
    ) -> Float[Array, "seq state_dim"]:
        dtype = x_t.dtype
        dt = jax.nn.softplus(self.dt_bias).astype(dtype)
        A = jnp.exp(self.A_log).astype(dtype)
        decay = jnp.exp(-dt * A)
        injected = jnp.einsum("...i,oi->...o", e, self.B.astype(e.dtype)).astype(dtype)
        return x_t * decay + dt * injected

    def spectral_norm(self) -> Array:
        dt = jax.nn.softplus(self.dt_bias)
        A = jnp.exp(self.A_log)
        return jnp.max(jnp.exp(-dt * A))

    def contraction_factor(self) -> Array:
        dt = jax.nn.softplus(self.dt_bias)
        A = jnp.exp(self.A_log)
        return jnp.mean(jnp.exp(-dt * A))


class VisionParcaeDiagonalExactZOHInjection(VisionParcaeDiagonalInjection):
    """Parcae diagonal injection with an exact zero-order-hold input write.

        x[t + 1] = exp(-softplus(dt) * exp(A_log)) * x[t]
                 + ((1 - exp(-softplus(dt) * exp(A_log))) / exp(A_log)) * B e

    This keeps the same diagonal state decay, ``A_log``/``dt_bias``
    parameterisation, full ``B`` write matrix, and identity-padded ``B``
    initialisation as ``VisionParcaeDiagonalInjection``.  It changes only the
    gain applied to the full-matrix input write.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        *,
        dt_init: float = 0.1,
        A_init: float = 1.0,
        B_init_scale: float = 1.0,
        key: PRNGKeyArray | None = None,
    ):
        super().__init__(
            input_dim,
            state_dim,
            dt_init=dt_init,
            A_init=A_init,
            B_init_scale=B_init_scale,
            key=key,
        )

    def __call__(
        self,
        x_t: Float[Array, "seq state_dim"],
        e: Float[Array, "seq input_dim"],
    ) -> Float[Array, "seq state_dim"]:
        dtype = x_t.dtype
        compute_dtype = jnp.result_type(self.dt_bias, self.A_log, jnp.float32)
        dt = jax.nn.softplus(self.dt_bias.astype(compute_dtype))
        A = jnp.exp(self.A_log.astype(compute_dtype))
        decay = jnp.exp(-dt * A).astype(dtype)
        input_gain = _exact_zoh_input_gain(dt, A).astype(dtype)
        injected = jnp.einsum("...i,oi->...o", e, self.B.astype(e.dtype)).astype(dtype)
        return x_t * decay + input_gain * injected


class VisionParcaeLinearInjection(eqx.Module):
    """Concatenate recurrent state and prelude embeddings, then project."""

    adapter: eqx.nn.Linear

    input_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        *,
        bias: bool = True,
        init_std: float | None = None,
        key: PRNGKeyArray,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        if init_std is None:
            init_std = _takase_std(state_dim + input_dim)
        self.adapter = _make_scaled_linear(
            state_dim + input_dim,
            state_dim,
            use_bias=bias,
            key=key,
            std=init_std,
        )

    def __call__(
        self,
        x_t: Float[Array, "seq state_dim"],
        e: Float[Array, "seq input_dim"],
    ) -> Float[Array, "seq state_dim"]:
        return _apply_linear_sequence(self.adapter, jnp.concatenate([x_t, e], axis=-1))

    def spectral_norm(self) -> Array:
        # Recurrent-state slice of W[state, input] is the linearized A term.
        return jnp.linalg.svd(
            self.adapter.weight[:, : self.state_dim].astype(jnp.float32),
            compute_uv=False,
        )[0]

    def contraction_factor(self) -> Array:
        return self.spectral_norm()


class VisionParcaeAdditiveInjection(eqx.Module):
    """Simple additive injection.  Requires equal state and input dimensions."""

    input_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)

    def __init__(self, input_dim: int, state_dim: int):
        if input_dim != state_dim:
            raise ValueError(
                "Additive injection requires recurrent_dim == dim; "
                f"got recurrent_dim={state_dim}, dim={input_dim}."
            )
        self.input_dim = input_dim
        self.state_dim = state_dim

    def __call__(
        self,
        x_t: Float[Array, "seq state_dim"],
        e: Float[Array, "seq input_dim"],
    ) -> Float[Array, "seq state_dim"]:
        return x_t + e

    def spectral_norm(self) -> Array:
        return jnp.asarray(1.0, dtype=jnp.float32)

    def contraction_factor(self) -> Array:
        return jnp.asarray(1.0, dtype=jnp.float32)


def _get_vision_parcae_injection(
    injection_type: InjectionKind,
    *,
    input_dim: int,
    state_dim: int,
    linear_bias: bool,
    linear_init_std: float | None,
    dt_init: float,
    A_init: float,
    B_init_scale: float,
    key: PRNGKeyArray,
) -> eqx.Module:
    if injection_type == "diagonal":
        return VisionParcaeDiagonalInjection(
            input_dim,
            state_dim,
            dt_init=dt_init,
            A_init=A_init,
            B_init_scale=B_init_scale,
            key=key,
        )
    if injection_type == "diagonal_exact_zoh":
        return VisionParcaeDiagonalExactZOHInjection(
            input_dim,
            state_dim,
            dt_init=dt_init,
            A_init=A_init,
            B_init_scale=B_init_scale,
            key=key,
        )
    if injection_type == "linear":
        return VisionParcaeLinearInjection(
            input_dim,
            state_dim,
            bias=linear_bias,
            init_std=linear_init_std,
            key=key,
        )
    if injection_type == "add":
        return VisionParcaeAdditiveInjection(input_dim, state_dim)
    raise ValueError(f"Invalid injection_type: {injection_type!r}.")


@register_model("vision_parcae", modality="vision")
class VisionParcae(eqx.Module):
    """Vision Parcae: a ViT front/end with a stable looped transformer core.

    The model follows the same unbatched image convention as Equimo's ViT:
    inputs have shape ``(channels, height, width)`` and batching is expected to
    be done by ``jax.vmap`` outside the module.  With ``sample_recurrence=True``,
    a vmapped call samples a different depth for each image while using one
    static ``lax.scan`` bound, so the recurrent path remains JIT-compatible.

    A few deliberate-but-non-obvious properties of this implementation:

    * **Per-image variable depth via ``vmap`` does not save FLOPs.**  Under
      ``vmap`` the ``lax.cond`` gating each iteration becomes ``lax.select``,
      so both the active and inactive branches run on every image at every
      iteration regardless of the sampled depth.  Expected wall-clock cost is
      ``max_recurrence``-step training, not ``mean_recurrence``-step training.
      Pick ``max_recurrence`` close to the largest depth you actually need
      (defaults to ``2 * mean_recurrence`` or a 4σ Poisson cap).
    * **Out-of-range step counts are clipped silently for tracers and raise
      eagerly for Python ints.**  ``num_steps`` / ``num_steps_pair`` passed as
      JAX scalars are clamped to ``max_recurrence`` rather than recompiling;
      the same values as Python ``int`` raise so ahead-of-time mistakes are
      caught.  Increase ``max_recurrence`` (and re-instantiate) for deeper
      test-time scaling.
    * **Recurrent diagnostics in the aux dict are detached.**  The
      ``x_recurrent_state``, ``x_recurrent_state_prev``, ``x_projected_recurrent``
      and ``recurrent_residual`` entries are wrapped in ``stop_gradient`` to
      mirror the reference's ``@torch.no_grad()`` ``monitor_module``, so they
      are safe to log but cannot be used for an auxiliary loss.  The main
      forward output (logits / features) carries gradients normally.
    """

    patch_embed: PatchEmbedding
    global_pos_embed: LearnedPosEmbed | None
    local_pos_embed: CompositeVisionRoPE | None
    recurrent_local_pos_embed: CompositeVisionRoPE | None
    cls_token: jax.Array | None
    reg_tokens: jax.Array | None
    mask_token: jax.Array | None

    prelude: BlockChunk | None
    adapter: eqx.Module
    core_block: BlockChunk | None
    C: eqx.nn.Linear
    coda: BlockChunk | None

    pos_drop: eqx.nn.Dropout
    prelude_norm: eqx.Module | None
    norm: eqx.Module
    local_cls_norm: eqx.Module | None
    head: eqx.Module

    dim: int = eqx.field(static=True)
    recurrent_dim: int = eqx.field(static=True)
    embed_size: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    global_pool: str = eqx.field(static=True)
    num_reg_tokens: int = eqx.field(static=True)
    num_prefix_tokens: int = eqx.field(static=True)
    num_embedded_prefix_tokens: int = eqx.field(static=True)
    global_pos_embed_cls: bool = eqx.field(static=True)
    global_pos_embed_reg: bool = eqx.field(static=True)
    local_pos_embed_reg: bool = eqx.field(static=True)
    embed_len: int = eqx.field(static=True)
    dynamic_img_size: bool = eqx.field(static=True)
    antialias: bool = eqx.field(static=True)

    n_layers_in_prelude: int = eqx.field(static=True)
    n_layers_in_recurrent_block: int = eqx.field(static=True)
    n_layers_in_coda: int = eqx.field(static=True)
    mean_recurrence: int = eqx.field(static=True)
    mean_backprop_depth: int = eqx.field(static=True)
    max_recurrence: int = eqx.field(static=True)
    injection_type: InjectionKind = eqx.field(static=True)
    B_init_mode: BInitMode = eqx.field(static=True)
    B_init_target_depth: int | None = eqx.field(static=True)
    B_init_target_scale: float = eqx.field(static=True)
    state_init: StateInitKind = eqx.field(static=True)
    state_init_scale: float | None = eqx.field(static=True)
    sample_recurrence: bool = eqx.field(static=True)
    sampling_scheme: SamplingKind = eqx.field(static=True)
    recurrent_checkpoint: bool = eqx.field(static=True)
    coda_checkpoint: bool = eqx.field(static=True)
    C_init: CInitKind = eqx.field(static=True)

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        dim: int,
        patch_size: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        n_layers_in_prelude: int = 2,
        n_layers_in_recurrent_block: int = 2,
        n_layers_in_coda: int = 2,
        mean_recurrence: int = 8,
        mean_backprop_depth: int | None = None,
        max_recurrence: int | None = None,
        recurrent_dim: int | None = None,
        recurrent_num_heads: int | None = None,
        injection_type: InjectionKind = "diagonal",
        state_init: StateInitKind = "like-init",
        state_init_scale: float | None = None,
        prelude_norm: bool = True,
        sample_recurrence: bool = False,
        sampling_scheme: SamplingKind = "fixed",
        recurrent_checkpoint: bool = False,
        coda_checkpoint: bool = False,
        use_mask_token: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        class_token: bool = True,
        global_pos_embed_cls: bool = True,
        global_pos_embed_reg: bool = False,
        local_pos_embed_reg: bool = False,
        reg_tokens: int = 0,
        use_global_pos_embed: bool = True,
        use_local_pos_embed: bool = False,
        local_pos_embed_config_patch: dict = {
            "strategy": "period",
            "base": 100.0,
            "normalize_coords": "separate",
            "dtype": jnp.float32,
        },
        local_pos_embed_config_reg: dict = {
            "strategy": "period",
            "base": 100.0,
            "normalize_coords": "separate",
            "dtype": jnp.float32,
        },
        pos_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        block: str | type[eqx.Module] = "attentionblock",
        mlp_ratio: float = 4.0,
        recurrent_mlp_ratio: float | None = None,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        recurrent_qk_norm: bool | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: str | Callable = "gelu",
        attn_layer: str | type[eqx.Module] = "attention",
        ffn_layer: str | type[eqx.Module] = "mlp",
        ffn_bias: bool = True,
        ffn_kwargs: dict = {},
        recurrent_ffn_kwargs: dict | None = None,
        norm_layer: str | type[eqx.Module] = "layernorm",
        untie_global_and_local_cls_norm: bool = False,
        init_values: float | None = None,
        recurrent_init_values: float | None = None,
        global_pool: Literal["", "token", "avg", "avgmax", "max"] = "avg",
        num_classes: int | None = 1000,
        interpolate_antialias: bool = False,
        eps: float = 1e-5,
        dt_init: float = 0.1,
        alpha_init: float | None = None,
        B_init_mode: BInitMode = "fixed_point",
        B_init_target_depth: int | None = None,
        B_init_target_scale: float = 1.0,
        A_init: float | None = None,
        B_init_scale: float | None = None,
        linear_injection_bias: bool = True,
        injection_bias: bool | None = None,
        linear_injection_init_std: float | None = None,
        C_bias: bool = True,
        C_init: CInitKind = "scaled",
        C_init_std: float | None = None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unexpected VisionParcae arguments: {sorted(kwargs)}")

        if injection_bias is not None:
            if injection_type != "linear":
                raise ValueError(
                    "injection_bias only applies to injection_type='linear'. "
                    "Use linear_injection_bias for new code."
                )
            linear_injection_bias = injection_bias

        if n_layers_in_recurrent_block <= 0:
            raise ValueError("VisionParcae requires n_layers_in_recurrent_block > 0.")
        if mean_recurrence < 1:
            raise ValueError("mean_recurrence must be >= 1.")

        if dt_init <= 0:
            raise ValueError("dt_init must be positive.")
        if B_init_mode not in ("raw", "fixed_point", "target_depth", "one_step"):
            raise ValueError(f"Unknown B init mode: {B_init_mode!r}.")

        if B_init_mode == "fixed_point" and B_init_target_depth is not None:
            raise ValueError(
                "B_init_target_depth is only used with B_init_mode='target_depth'."
            )

        if B_init_mode == "one_step" and B_init_target_depth is not None:
            raise ValueError("B_init_mode='one_step' does not use B_init_target_depth.")

        resolved_B_init_mode = B_init_mode
        if alpha_init is not None:
            if A_init is not None:
                raise ValueError("alpha_init and A_init are mutually exclusive.")
            if B_init_scale is not None and B_init_mode != "raw":
                raise ValueError(
                    "B_init_scale with alpha_init requires B_init_mode='raw'."
                )
            if B_init_mode == "raw" and (
                B_init_target_depth is not None or B_init_target_scale != 1.0
            ):
                raise ValueError("raw B_init_mode does not use target settings.")

            is_diagonal_injection = injection_type in ("diagonal", "diagonal_exact_zoh")
            if not is_diagonal_injection:
                raise ValueError("alpha_init only applies to diagonal injections.")

            A_init, B_init_scale = dynamics_from_alpha(
                alpha=alpha_init,
                dt=dt_init,
                injection_type=injection_type,
                mode=B_init_mode,
                target_depth=B_init_target_depth,
                target_scale=B_init_target_scale,
                raw_B_init_scale=B_init_scale,
            )
        elif B_init_mode in ("target_depth", "one_step"):
            raise ValueError(f"B_init_mode={B_init_mode!r} requires alpha_init.")
        elif B_init_target_depth is not None or B_init_target_scale != 1.0:
            raise ValueError("B_init target settings require alpha_init.")
        else:
            resolved_B_init_mode = "raw"

        A_init = 1.0 if A_init is None else A_init
        B_init_scale = 1.0 if B_init_scale is None else B_init_scale

        recurrent_dim = dim if recurrent_dim is None else recurrent_dim
        recurrent_num_heads = (
            num_heads if recurrent_num_heads is None else recurrent_num_heads
        )
        recurrent_mlp_ratio = (
            mlp_ratio if recurrent_mlp_ratio is None else recurrent_mlp_ratio
        )
        recurrent_qk_norm = qk_norm if recurrent_qk_norm is None else recurrent_qk_norm
        recurrent_ffn_kwargs = (
            ffn_kwargs if recurrent_ffn_kwargs is None else recurrent_ffn_kwargs
        )
        recurrent_init_values = (
            init_values if recurrent_init_values is None else recurrent_init_values
        )
        mean_backprop_depth = (
            (mean_recurrence + 1) // 2
            if mean_backprop_depth is None
            else mean_backprop_depth
        )
        if mean_backprop_depth < 0:
            raise ValueError("mean_backprop_depth must be >= 0.")
        if C_init not in ("scaled", "identity"):
            raise ValueError(f"Invalid C_init: {C_init!r}.")

        if max_recurrence is None:
            max_recurrence = max(
                _default_max_recurrence(mean_recurrence, sample_recurrence),
                mean_backprop_depth,
            )
        if max_recurrence < mean_recurrence:
            raise ValueError("max_recurrence must be >= mean_recurrence.")
        if max_recurrence < mean_backprop_depth:
            raise ValueError("max_recurrence must be >= mean_backprop_depth.")
        if max_recurrence < 1:
            raise ValueError("max_recurrence must be >= 1.")

        effective_adapter_depth = (
            n_layers_in_prelude + n_layers_in_recurrent_block * mean_recurrence
        )
        if linear_injection_init_std is None:
            linear_injection_init_std = _depth_scaled_std(
                recurrent_dim + dim,
                effective_adapter_depth,
            )
        if C_init_std is None:
            C_init_std = _depth_scaled_std(recurrent_dim, n_layers_in_prelude)

        (
            key_patchemb,
            key_posemb,
            key_cls,
            key_reg,
            key_prelude,
            key_adapter,
            key_state_project,
            key_core,
            key_coda,
            key_head,
        ) = jr.split(key, 10)

        self.dim = dim
        self.recurrent_dim = recurrent_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.num_embedded_prefix_tokens = 0
        self.dynamic_img_size = dynamic_img_size
        self.antialias = interpolate_antialias
        self.global_pos_embed_cls = global_pos_embed_cls
        self.global_pos_embed_reg = global_pos_embed_reg
        self.local_pos_embed_reg = local_pos_embed_reg
        self.global_pool = global_pool
        self.embed_size = img_size // patch_size
        self.n_layers_in_prelude = n_layers_in_prelude
        self.n_layers_in_recurrent_block = n_layers_in_recurrent_block
        self.n_layers_in_coda = n_layers_in_coda
        self.mean_recurrence = mean_recurrence
        self.mean_backprop_depth = mean_backprop_depth
        self.max_recurrence = max_recurrence
        self.injection_type = injection_type
        self.B_init_mode = resolved_B_init_mode
        self.B_init_target_depth = B_init_target_depth
        self.B_init_target_scale = B_init_target_scale
        self.state_init = state_init
        self.state_init_scale = state_init_scale
        self.sample_recurrence = sample_recurrence
        self.sampling_scheme = sampling_scheme
        self.recurrent_checkpoint = recurrent_checkpoint
        self.coda_checkpoint = coda_checkpoint
        self.C_init = C_init

        block = get_attn_block(block)
        attn_layer = get_attn(attn_layer)
        ffn_layer = get_ffn(ffn_layer)
        norm_layer = get_norm(norm_layer)
        act_layer = get_act(act_layer)

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=dim,
            patch_size=patch_size,
            img_size=img_size,
            flatten=not dynamic_img_size,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            key=key_patchemb,
        )
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = jr.normal(key_cls, (1, dim)) if class_token else None
        self.reg_tokens = (
            jr.normal(key_reg, (reg_tokens, dim)) if reg_tokens > 0 else None
        )
        self.mask_token = jnp.zeros((1, dim)) if use_mask_token else None

        if not global_pos_embed_cls:
            self.embed_len = self.num_patches
        elif global_pos_embed_reg:
            self.embed_len = self.num_patches + self.num_prefix_tokens
            self.num_embedded_prefix_tokens += self.num_prefix_tokens
        else:
            self.num_embedded_prefix_tokens += 1
            self.embed_len = self.num_patches + 1

        if use_global_pos_embed:
            self.global_pos_embed = LearnedPosEmbed(
                weight=jr.normal(key_posemb, (self.embed_len, dim)),
                dim=dim,
                embed_size=self.embed_size,
                num_prefix_tokens=self.num_prefix_tokens,
                num_embedded_prefix_tokens=self.num_embedded_prefix_tokens,
                global_pos_embed_cls=global_pos_embed_cls,
                global_pos_embed_reg=global_pos_embed_reg,
                antialias=interpolate_antialias,
            )
        else:
            self.global_pos_embed = None

        def build_local_pos_embed(
            *, dim_: int, num_heads_: int
        ) -> CompositeVisionRoPE | None:
            if not use_local_pos_embed:
                return None
            if not isinstance(num_heads_, int):
                raise ValueError(
                    "VisionParcae local RoPE requires static integer heads."
                )
            patch_rope = VisionRoPE(
                dim=dim_,
                num_heads=num_heads_,
                **local_pos_embed_config_patch,
            )
            n_prefix = (
                (1 if class_token else 0)
                if local_pos_embed_reg
                else self.num_prefix_tokens
            )
            n_reg = self.num_reg_tokens if local_pos_embed_reg else 0
            reg_rope = (
                VisionRoPE(
                    dim=dim_,
                    num_heads=num_heads_,
                    **local_pos_embed_config_reg,
                )
                if n_reg > 0
                else None
            )
            return CompositeVisionRoPE(
                patch_rope,
                reg_rope=reg_rope,
                num_prefix_tokens=n_prefix,
                num_registers=n_reg,
            )

        self.local_pos_embed = build_local_pos_embed(dim_=dim, num_heads_=num_heads)
        self.recurrent_local_pos_embed = (
            self.local_pos_embed
            if recurrent_dim == dim and recurrent_num_heads == num_heads
            else build_local_pos_embed(
                dim_=recurrent_dim, num_heads_=recurrent_num_heads
            )
        )

        self.pos_drop = eqx.nn.Dropout(pos_drop_rate)

        # Drop-path is assigned over physical blocks.  The recurrent chunk reuses
        # its physical rates each loop iteration, which mirrors weight sharing.
        physical_depth = (
            n_layers_in_prelude + n_layers_in_recurrent_block + n_layers_in_coda
        )
        if drop_path_uniform:
            dpr = [drop_path_rate] * physical_depth
        else:
            dpr = np.linspace(0.0, drop_path_rate, physical_depth).tolist()
        prelude_dpr = dpr[:n_layers_in_prelude]
        core_start = n_layers_in_prelude
        core_end = core_start + n_layers_in_recurrent_block
        core_dpr = dpr[core_start:core_end]
        coda_dpr = dpr[core_end:]

        self.prelude = _make_block_chunk(
            depth=n_layers_in_prelude,
            dim=dim,
            num_heads=num_heads,
            block=block,
            attn_layer=attn_layer,
            ffn_layer=ffn_layer,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            act_layer=act_layer,
            ffn_bias=ffn_bias,
            ffn_kwargs=ffn_kwargs,
            norm_layer=norm_layer,
            eps=eps,
            drop_path=prelude_dpr,
            init_values=init_values,
            key=key_prelude,
        )

        self.adapter = _get_vision_parcae_injection(
            injection_type,
            input_dim=dim,
            state_dim=recurrent_dim,
            linear_bias=linear_injection_bias,
            linear_init_std=linear_injection_init_std,
            dt_init=dt_init,
            A_init=A_init,
            B_init_scale=B_init_scale,
            key=key_adapter,
        )

        self.core_block = _make_block_chunk(
            depth=n_layers_in_recurrent_block,
            dim=recurrent_dim,
            num_heads=recurrent_num_heads,
            block=block,
            attn_layer=attn_layer,
            ffn_layer=ffn_layer,
            mlp_ratio=recurrent_mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=recurrent_qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            act_layer=act_layer,
            ffn_bias=ffn_bias,
            ffn_kwargs=recurrent_ffn_kwargs,
            norm_layer=norm_layer,
            eps=eps,
            drop_path=core_dpr,
            init_values=recurrent_init_values,
            key=key_core,
        )

        if C_init == "identity":
            self.C = _make_identity_or_padded_linear(
                recurrent_dim,
                dim,
                use_bias=C_bias,
                key=key_state_project,
            )
        else:
            self.C = _make_scaled_linear(
                recurrent_dim,
                dim,
                use_bias=C_bias,
                key=key_state_project,
                std=C_init_std,
            )

        self.coda = _make_block_chunk(
            depth=n_layers_in_coda,
            dim=dim,
            num_heads=num_heads,
            block=block,
            attn_layer=attn_layer,
            ffn_layer=ffn_layer,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            act_layer=act_layer,
            ffn_bias=ffn_bias,
            ffn_kwargs=ffn_kwargs,
            norm_layer=norm_layer,
            eps=eps,
            drop_path=coda_dpr,
            init_values=init_values,
            key=key_coda,
        )

        self.prelude_norm = norm_layer(dim, eps=eps) if prelude_norm else None
        self.norm = norm_layer(dim, eps=eps)
        self.local_cls_norm = (
            norm_layer(dim, eps=eps) if untie_global_and_local_cls_norm else None
        )
        self.head = (
            eqx.nn.Linear(dim, num_classes, key=key_head)
            if num_classes is not None and num_classes > 0
            else eqx.nn.Identity()
        )

    def _embed_image(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: PRNGKeyArray,
        mask: Optional[Int[Array, "embed_h embed_w"]] = None,
        inference: Optional[bool] = None,
    ) -> tuple[Float[Array, "seq dim"], int, int]:
        x = self.patch_embed(x)
        if self.dynamic_img_size:
            _, H, W = x.shape
        else:
            H = W = self.embed_size

        if mask is not None:
            if self.mask_token is None:
                raise ValueError(
                    "To use masked forward, initialise the model with use_mask_token=True."
                )
            if self.dynamic_img_size:
                mask = rearrange(mask, "h w -> 1 h w")
                value = rearrange(self.mask_token, "1 c -> c 1 1")
            else:
                mask = rearrange(mask, "h w -> (h w) 1")
                value = self.mask_token
            x = jnp.where(mask, x, value.astype(x.dtype))

        if self.global_pos_embed is not None:
            x = self.global_pos_embed(
                x,
                cls_token=self.cls_token,
                reg_tokens=self.reg_tokens,
                dynamic_img_size=self.dynamic_img_size,
            )
        else:
            prefix = [t for t in (self.cls_token, self.reg_tokens) if t is not None]
            if self.dynamic_img_size:
                x = rearrange(x, "c h w -> (h w) c")
            x = jnp.concatenate([*prefix, x], axis=0) if prefix else x

        x = self.pos_drop(x, inference=inference, key=key)
        return x, H, W

    def _rope_sincos(
        self,
        pos_embed: CompositeVisionRoPE | None,
        *,
        H: int,
        W: int,
        inference: Optional[bool],
        key: PRNGKeyArray,
    ):
        if pos_embed is None:
            return None
        return pos_embed.get_sincos(H=H, W=W, inference=inference, key=key)

    def _run_chunk(
        self,
        chunk: BlockChunk | None,
        x: Float[Array, "seq dim"],
        *,
        pos_embed: CompositeVisionRoPE | None,
        H: int,
        W: int,
        inference: Optional[bool],
        key: PRNGKeyArray,
        rope_sincos=None,
        checkpoint: bool = False,
        **kwargs,
    ):
        if chunk is None:
            return x
        key_rope, key_chunk = jr.split(key, 2)
        if rope_sincos is None:
            rope_sincos = self._rope_sincos(
                pos_embed,
                H=H,
                W=W,
                inference=inference,
                key=key_rope,
            )

        def run(y):
            return chunk(
                y,
                rope_sincos=rope_sincos,
                inference=inference,
                key=key_chunk,
                **kwargs,
            )

        return eqx.filter_checkpoint(run)(x) if checkpoint else run(x)

    def _initialize_state(
        self,
        e: Float[Array, "seq dim"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "seq recurrent_dim"]:
        shape = e.shape[:-1] + (self.recurrent_dim,)
        dtype = e.dtype

        if self.state_init == "zero":
            return jnp.zeros(shape, dtype=dtype)
        if self.state_init == "normal":
            std = 1.0 if self.state_init_scale is None else self.state_init_scale
            return (jr.normal(key, shape) * jnp.asarray(std, dtype=dtype)).astype(dtype)
        if self.state_init == "embed":
            scale = 1.0 / jnp.sqrt(jnp.asarray(e.shape[-1], dtype=dtype))
            return (jr.normal(key, shape) * scale).astype(dtype)
        if self.state_init == "unit":
            z = jr.normal(key, shape)
            z = (z - z.mean(axis=-1, keepdims=True)) / (
                z.std(axis=-1, keepdims=True) + 1e-8
            )
            return z.astype(dtype)
        if self.state_init == "like-init":
            std = (
                _takase_std(self.dim)
                if self.state_init_scale is None
                else self.state_init_scale
            )
            return (
                jr.truncated_normal(key, lower=-3.0, upper=3.0, shape=shape)
                * jnp.asarray(std, dtype=dtype)
            ).astype(dtype)

        raise ValueError(f"Invalid state_init: {self.state_init!r}.")

    def _clip_step_pair(
        self,
        no_grad_steps: Array,
        grad_steps: Array,
    ) -> tuple[Array, Array]:
        no_grad_steps = jnp.maximum(no_grad_steps.astype(jnp.int32), 0)
        grad_steps = jnp.maximum(grad_steps.astype(jnp.int32), 0)
        grad_steps = jnp.minimum(
            grad_steps, jnp.asarray(self.max_recurrence, jnp.int32)
        )
        no_grad_steps = jnp.minimum(
            no_grad_steps,
            jnp.asarray(self.max_recurrence, jnp.int32) - grad_steps,
        )
        return no_grad_steps, grad_steps

    def sample_num_steps(
        self,
        key: PRNGKeyArray,
        *,
        inference: Optional[bool] = None,
    ) -> tuple[Array, Array]:
        """Sample ``(no_grad_steps, grad_steps)`` as JAX scalars.

        The returned values are arrays rather than Python ints, so this method
        is safe under ``jax.jit``/``eqx.filter_jit``.  Poisson draws are clipped
        to the static ``max_recurrence`` scan bound.

        Note that under ``vmap`` the per-image sampled depth selects which
        carry to keep but does not skip iterations: every image runs the full
        ``max_recurrence`` loop because ``lax.cond`` becomes ``lax.select``
        when its predicate is batched.  See the class docstring for details.
        """

        if inference:
            return self._clip_step_pair(
                jnp.asarray(0, jnp.int32),
                jnp.asarray(self.mean_recurrence, jnp.int32),
            )

        no_grad_mean = max(self.mean_recurrence - self.mean_backprop_depth, 0)
        grad_mean = self.mean_backprop_depth

        if self.sampling_scheme == "fixed":
            return self._clip_step_pair(
                jnp.asarray(no_grad_mean, jnp.int32),
                jnp.asarray(grad_mean, jnp.int32),
            )

        key_n = key
        if self.sampling_scheme == "poisson-truncated-full":
            # Paper/per-sequence semantics: sample total T, then split off the
            # gradient suffix k = min(T, mu_bwd).  The PyTorch per-batch helper
            # differs, but its per-sequence helper uses this version.
            total = jr.poisson(
                key_n,
                jnp.asarray(self.mean_recurrence, jnp.float32),
                shape=(),
            ).astype(jnp.int32)
            total = jnp.clip(total, 1, self.max_recurrence)
            grad = jnp.minimum(total, jnp.asarray(grad_mean, jnp.int32))
            return total - grad, grad

        if self.sampling_scheme == "poisson-full":
            total = jr.poisson(
                key_n,
                jnp.asarray(self.mean_recurrence, jnp.float32),
                shape=(),
            ).astype(jnp.int32)
            total = jnp.clip(total, 1, self.max_recurrence)
            return jnp.asarray(0, jnp.int32), total

        if self.sampling_scheme == "poisson-fill":
            no_grad = jr.poisson(
                key_n,
                jnp.asarray(no_grad_mean, jnp.float32),
                shape=(),
            ).astype(jnp.int32)
            return self._clip_step_pair(no_grad, jnp.asarray(grad_mean, jnp.int32))

        raise ValueError(f"Invalid sampling_scheme: {self.sampling_scheme!r}.")

    def _resolve_num_steps(
        self,
        *,
        num_steps: int | Array | None,
        num_steps_pair: tuple[int | Array, int | Array]
        | list[int | Array]
        | Array
        | None,
        inference: Optional[bool],
        key: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        if num_steps_pair is not None:
            if isinstance(num_steps_pair, (tuple, list)):
                if len(num_steps_pair) != 2:
                    raise ValueError(
                        "num_steps_pair must be a pair: (no_grad_steps, grad_steps)."
                    )
                no_grad = _to_scalar_i32(num_steps_pair[0], name="num_steps_pair[0]")
                grad = _to_scalar_i32(num_steps_pair[1], name="num_steps_pair[1]")
                if all(isinstance(v, (int, np.integer)) for v in num_steps_pair):
                    if (
                        int(num_steps_pair[0]) + int(num_steps_pair[1])
                        > self.max_recurrence
                    ):
                        raise ValueError(
                            "num_steps_pair exceeds max_recurrence. Increase "
                            "max_recurrence when constructing the model."
                        )
            else:
                pair = jnp.asarray(num_steps_pair, dtype=jnp.int32)
                if pair.shape != (2,):
                    raise ValueError("num_steps_pair must have shape (2,).")
                no_grad, grad = pair[0], pair[1]
            return self._clip_step_pair(no_grad, grad)

        if num_steps is not None:
            if (
                isinstance(num_steps, (int, np.integer))
                and int(num_steps) > self.max_recurrence
            ):
                raise ValueError(
                    "num_steps exceeds max_recurrence. Increase max_recurrence "
                    "when constructing the model."
                )
            total = jnp.minimum(
                _to_scalar_i32(num_steps, name="num_steps"),
                jnp.asarray(self.max_recurrence, jnp.int32),
            )
            if inference:
                return jnp.asarray(0, jnp.int32), total
            grad = jnp.minimum(total, jnp.asarray(self.mean_backprop_depth, jnp.int32))
            return total - grad, grad

        if self.sample_recurrence and not inference:
            return self.sample_num_steps(key, inference=inference)

        if inference:
            return self._clip_step_pair(
                jnp.asarray(0, jnp.int32),
                jnp.asarray(self.mean_recurrence, jnp.int32),
            )
        grad = min(self.mean_recurrence, self.mean_backprop_depth)
        return self._clip_step_pair(
            jnp.asarray(self.mean_recurrence - grad, jnp.int32),
            jnp.asarray(grad, jnp.int32),
        )

    def _recurrent_step(
        self,
        h: Float[Array, "seq recurrent_dim"],
        e: Float[Array, "seq dim"],
        *,
        H: int,
        W: int,
        inference: Optional[bool],
        key: PRNGKeyArray,
        rope_sincos=None,
        checkpoint: bool = False,
        **kwargs,
    ) -> Float[Array, "seq recurrent_dim"]:
        h = self.adapter(h, e)
        h = self._run_chunk(
            self.core_block,
            h,
            pos_embed=self.recurrent_local_pos_embed,
            H=H,
            W=W,
            inference=inference,
            key=key,
            rope_sincos=rope_sincos,
            checkpoint=checkpoint,
            **kwargs,
        )
        return h

    def _iterate_recurrent(
        self,
        e: Float[Array, "seq dim"],
        *,
        H: int,
        W: int,
        inference: Optional[bool],
        key: PRNGKeyArray,
        num_steps: int | Array | None = None,
        num_steps_pair: tuple[int | Array, int | Array]
        | list[int | Array]
        | Array
        | None = None,
        **kwargs,
    ) -> tuple[Float[Array, "seq recurrent_dim"], dict]:
        key_state, key_steps, key_loop, key_rope = jr.split(key, 4)
        h = self._initialize_state(e, key=key_state)
        no_grad_steps, grad_steps = self._resolve_num_steps(
            num_steps=num_steps,
            num_steps_pair=num_steps_pair,
            inference=inference,
            key=key_steps,
        )
        total_steps = no_grad_steps + grad_steps

        rope_sincos = self._rope_sincos(
            self.recurrent_local_pos_embed,
            H=H,
            W=W,
            inference=inference,
            key=key_rope,
        )

        keys = jr.split(key_loop, self.max_recurrence)

        # Keep the detached prefix out of the reverse-mode scan. With batched
        # stochastic depths, masking no-grad steps inside one scan still makes
        # XLA carry the full max_recurrence body through the backward pass.
        if inference:
            no_grad_scan_bound = 0
            grad_scan_bound = self.max_recurrence
        elif num_steps_pair is not None:
            no_grad_scan_bound = self.max_recurrence
            grad_scan_bound = self.max_recurrence
        elif (
            num_steps is None
            and self.sample_recurrence
            and self.sampling_scheme == "poisson-full"
        ):
            no_grad_scan_bound = 0
            grad_scan_bound = self.max_recurrence
        else:
            grad_scan_bound = min(self.max_recurrence, self.mean_backprop_depth)
            no_grad_scan_bound = self.max_recurrence - grad_scan_bound

        def no_grad_body(carry, xs):
            h_cur, last_cur = carry
            t, key_step = xs
            active = t < no_grad_steps

            def inactive(c):
                return c

            def active_update(c):
                h_active, _ = c
                h_next = self._recurrent_step(
                    jax.lax.stop_gradient(h_active),
                    jax.lax.stop_gradient(e),
                    H=H,
                    W=W,
                    inference=inference,
                    key=key_step,
                    rope_sincos=rope_sincos,
                    checkpoint=False,
                    **kwargs,
                )
                return jax.lax.stop_gradient(h_next), h_active

            return jax.lax.cond(
                active, active_update, inactive, (h_cur, last_cur)
            ), None

        if no_grad_scan_bound:
            no_grad_ts = jnp.arange(no_grad_scan_bound, dtype=jnp.int32)
            (h, last_h), _ = jax.lax.scan(
                no_grad_body,
                (h, h),
                (no_grad_ts, keys[:no_grad_scan_bound]),
            )
            h = jax.lax.stop_gradient(h)
            last_h = jax.lax.stop_gradient(last_h)
        else:
            last_h = h

        def grad_body(carry, t):
            h_cur, last_cur = carry
            active = t < grad_steps

            def inactive(c):
                return c

            def active_update(c):
                h_active, _ = c
                key_index = jnp.minimum(
                    no_grad_steps + t,
                    jnp.asarray(self.max_recurrence - 1, jnp.int32),
                )
                key_step = jax.lax.dynamic_index_in_dim(
                    keys, key_index, axis=0, keepdims=False
                )
                h_next = self._recurrent_step(
                    h_active,
                    e,
                    H=H,
                    W=W,
                    inference=inference,
                    key=key_step,
                    rope_sincos=rope_sincos,
                    checkpoint=self.recurrent_checkpoint,
                    **kwargs,
                )
                return h_next, h_active

            return jax.lax.cond(
                active, active_update, inactive, (h_cur, last_cur)
            ), None

        if grad_scan_bound:
            grad_ts = jnp.arange(grad_scan_bound, dtype=jnp.int32)
            (h, last_h), _ = jax.lax.scan(grad_body, (h, last_h), grad_ts)

        residual = jnp.mean(jnp.linalg.norm(h - last_h, axis=-1))
        aux = {
            "num_steps": total_steps,
            "num_steps_no_grad": no_grad_steps,
            "num_steps_with_grad": grad_steps,
            "recurrent_residual": jax.lax.stop_gradient(residual),
            "x_recurrent_state": jax.lax.stop_gradient(h),
            "x_recurrent_state_prev": jax.lax.stop_gradient(last_h),
        }
        return h, aux

    def _features_with_aux(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: PRNGKeyArray,
        mask: Optional[Int[Array, "embed_h embed_w"]] = None,
        inference: Optional[bool] = None,
        num_steps: int | Array | None = None,
        num_steps_pair: tuple[int | Array, int | Array]
        | list[int | Array]
        | Array
        | None = None,
        **kwargs,
    ) -> tuple[Float[Array, "seq dim"], dict]:
        key_embed, key_prelude, key_loop, key_coda = jr.split(key, 4)

        x, H, W = self._embed_image(x, key=key_embed, mask=mask, inference=inference)
        x = self._run_chunk(
            self.prelude,
            x,
            pos_embed=self.local_pos_embed,
            H=H,
            W=W,
            inference=inference,
            key=key_prelude,
            **kwargs,
        )

        e = (
            _apply_norm_sequence(self.prelude_norm, x)
            if self.prelude_norm is not None
            else x
        )
        h, aux = self._iterate_recurrent(
            e,
            H=H,
            W=W,
            inference=inference,
            key=key_loop,
            num_steps=num_steps,
            num_steps_pair=num_steps_pair,
            **kwargs,
        )

        x = _apply_linear_sequence(self.C, h)
        aux["x_projected_recurrent"] = jax.lax.stop_gradient(x)

        x = self._run_chunk(
            self.coda,
            x,
            pos_embed=self.local_pos_embed,
            H=H,
            W=W,
            inference=inference,
            key=key_coda,
            checkpoint=self.coda_checkpoint,
            **kwargs,
        )
        return x, aux

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        mask: Optional[Int[Array, "embed_h embed_w"]] = None,
        inference: Optional[bool] = None,
        num_steps: int | Array | None = None,
        num_steps_pair: tuple[int | Array, int | Array]
        | list[int | Array]
        | Array
        | None = None,
        **kwargs,
    ) -> Float[Array, "seq dim"]:
        """Extract pre-classifier token features after Parcae looping.

        See ``__call__`` for the semantics of ``num_steps`` and
        ``num_steps_pair`` (Python ints validated eagerly; JAX scalars clipped
        silently to ``max_recurrence``).
        """

        x, _ = self._features_with_aux(
            x,
            key=key,
            mask=mask,
            inference=inference,
            num_steps=num_steps,
            num_steps_pair=num_steps_pair,
            **kwargs,
        )
        return x

    def forward_features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        """Return ViT-style token features plus Parcae recurrence diagnostics.

        Token outputs (``x_norm_cls_token``, ``x_norm_reg_tokens``,
        ``x_norm_patchtokens``, ``x_prenorm``) carry gradients normally.  The
        recurrent diagnostics included from the aux dict
        (``x_recurrent_state``, ``x_recurrent_state_prev``,
        ``x_projected_recurrent``, ``recurrent_residual``) are wrapped in
        ``stop_gradient`` and are intended for logging only — they cannot be
        used as auxiliary-loss targets.  This mirrors the reference's
        ``@torch.no_grad()`` ``monitor_module``.
        """

        x, aux = self._features_with_aux(x, key=key, inference=inference, **kwargs)
        x_norm = _apply_norm_sequence(self.norm, x)

        cls_offset = 1 if self.cls_token is not None else 0
        reg_start = cls_offset
        reg_end = reg_start + self.num_reg_tokens

        return {
            "x_norm_cls_token": x_norm[0] if self.cls_token is not None else None,
            "x_norm_reg_tokens": x_norm[reg_start:reg_end],
            "x_norm_patchtokens": x_norm[reg_end:],
            "x_prenorm": x,
            **aux,
        }

    def recurrent_spectral_norm(self) -> Array:
        """Return a spectral-norm diagnostic for the recurrent injection."""

        if hasattr(self.adapter, "spectral_norm"):
            return self.adapter.spectral_norm()
        return jnp.asarray(jnp.nan, dtype=jnp.float32)

    def recurrent_contraction_factor(self) -> Array:
        """Return a contraction diagnostic for the recurrent injection."""

        if hasattr(self.adapter, "contraction_factor"):
            return self.adapter.contraction_factor()
        return jnp.asarray(jnp.nan, dtype=jnp.float32)

    def C_spectral_norm(self) -> Array:
        """Spectral norm of the recurrent-state readout projection ``C``."""

        return jnp.linalg.svd(self.C.weight.astype(jnp.float32), compute_uv=False)[0]

    def weight_decay_mask(self) -> "VisionParcae":
        """Return a pytree mask for AdamW-style decoupled weight decay.

        ``True`` leaves receive weight decay.  The diagonal dynamical-system
        parameters ``A_log``, ``dt_bias``, ``B`` and the readout ``C.weight`` are
        marked ``False`` to match the Parcae reference implementation.
        """

        mask = jax.tree_util.tree_map(lambda _: True, self)
        mask = eqx.tree_at(lambda m: m.C.weight, mask, replace=False)
        if isinstance(self.adapter, VisionParcaeDiagonalInjection):
            mask = eqx.tree_at(
                lambda m: (m.adapter.A_log, m.adapter.dt_bias, m.adapter.B),
                mask,
                replace=(False, False, False),
            )
        return mask

    def _infer_hw_from_tokens(
        self,
        num_tokens: int,
        *,
        H: int | None = None,
        W: int | None = None,
    ) -> tuple[int, int]:
        patch_tokens = num_tokens - self.num_prefix_tokens
        if patch_tokens <= 0:
            raise ValueError("Cannot infer H/W from a state with no patch tokens.")
        if H is not None and W is not None:
            return H, W
        if H is not None:
            if patch_tokens % H != 0:
                raise ValueError(
                    "Cannot infer W; patch token count is not divisible by H."
                )
            return H, patch_tokens // H
        if W is not None:
            if patch_tokens % W != 0:
                raise ValueError(
                    "Cannot infer H; patch token count is not divisible by W."
                )
            return patch_tokens // W, W
        side = math.isqrt(patch_tokens)
        if side * side != patch_tokens:
            raise ValueError(
                "Cannot infer a square patch grid from the recurrent state. "
                "Pass H=... and W=... explicitly to readout."
            )
        return side, side

    def readout(
        self,
        h: Float[Array, "seq recurrent_dim"],
        *,
        H: int | None = None,
        W: int | None = None,
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = True,
        **kwargs,
    ) -> Float[Array, "num_classes"]:  # noqa: F821
        """Project a recurrent state through C, coda, final norm, pooling, and head."""

        if H is None or W is None:
            if self.dynamic_img_size or H is not None or W is not None:
                H, W = self._infer_hw_from_tokens(h.shape[0], H=H, W=W)
            else:
                H = self.embed_size
                W = self.embed_size

        x = _apply_linear_sequence(self.C, h)
        x = self._run_chunk(
            self.coda,
            x,
            pos_embed=self.local_pos_embed,
            H=H,
            W=W,
            inference=inference,
            key=key,
            checkpoint=self.coda_checkpoint,
            **kwargs,
        )
        x = _apply_norm_sequence(self.norm, x)
        x = pool_sd(
            x,
            num_prefix_tokens=self.num_prefix_tokens,
            pool_type=self.global_pool,
            reduce_include_prefix=False,
        )
        return self.head(x)

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        num_steps: int | Array | None = None,
        num_steps_pair: tuple[int | Array, int | Array]
        | list[int | Array]
        | Array
        | None = None,
        **kwargs,
    ) -> Float[Array, "num_classes"]:  # noqa: F821
        """Process an image and return classification logits.

        ``num_steps`` overrides the total recurrence depth and ``num_steps_pair``
        overrides the ``(no_grad, grad)`` split directly.  Either may be a
        Python ``int`` (validated eagerly — values exceeding ``max_recurrence``
        raise a ``ValueError``) or a JAX scalar (clipped silently to
        ``max_recurrence`` to avoid recompilation under ``jit``).  Increase
        ``max_recurrence`` and re-instantiate the model for deeper test-time
        scaling than was provisioned at construction.
        """

        x = self.features(
            x,
            key=key,
            inference=inference,
            num_steps=num_steps,
            num_steps_pair=num_steps_pair,
            **kwargs,
        )
        x = _apply_norm_sequence(self.norm, x)
        x = pool_sd(
            x,
            num_prefix_tokens=self.num_prefix_tokens,
            pool_type=self.global_pool,
            reduce_include_prefix=False,
        )
        return self.head(x)


_VISION_PARCAE_BASE_CFG: dict = {
    "img_size": 224,
    "in_channels": 3,
    "patch_size": 16,
    "num_classes": 1000,
    "reg_tokens": 0,
    "class_token": True,
    "use_mask_token": False,
    "dynamic_img_size": False,
    "act_layer": "gelu",
    "injection_type": "diagonal",
    "state_init": "like-init",
    "prelude_norm": True,
    "n_layers_in_prelude": 2,
    "n_layers_in_recurrent_block": 2,
    "n_layers_in_coda": 2,
    "mean_recurrence": 8,
    "mean_backprop_depth": 4,
    "sample_recurrence": False,
    "sampling_scheme": "fixed",
}

_VISION_PARCAE_REGISTRY: dict[str, tuple[dict, dict]] = {
    "vision_parcae_tiny_patch16_224": (
        _VISION_PARCAE_BASE_CFG,
        {"dim": 192, "num_heads": 3, "recurrent_dim": 192, "recurrent_num_heads": 3},
    ),
    "vision_parcae_small_patch16_224": (
        _VISION_PARCAE_BASE_CFG,
        {"dim": 384, "num_heads": 6, "recurrent_dim": 384, "recurrent_num_heads": 6},
    ),
    "vision_parcae_base_patch16_224": (
        _VISION_PARCAE_BASE_CFG,
        {"dim": 768, "num_heads": 12, "recurrent_dim": 768, "recurrent_num_heads": 12},
    ),
}


def _build_vision_parcae(
    variant: str,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> VisionParcae:
    """Build a VisionParcae variant from the local registry."""

    if key is None:
        key = jax.random.PRNGKey(42)
    if variant not in _VISION_PARCAE_REGISTRY:
        raise KeyError(f"Unknown VisionParcae variant: {variant!r}.")

    base_cfg, variant_cfg = _VISION_PARCAE_REGISTRY[variant]
    cfg = base_cfg | variant_cfg | overrides
    model = VisionParcae(**cfg, key=key)

    if pretrained:
        from equimo.serialization import load_weights

        model = load_weights(
            model,
            identifier=variant,
            inference_mode=inference_mode,
        )

    return model


def vision_parcae_tiny_patch16_224(**kwargs) -> VisionParcae:
    """Vision Parcae Ti/16 — 192-dim, 3 heads, 2+2+2 physical blocks."""

    return _build_vision_parcae("vision_parcae_tiny_patch16_224", **kwargs)


def vision_parcae_small_patch16_224(**kwargs) -> VisionParcae:
    """Vision Parcae S/16 — 384-dim, 6 heads, 2+2+2 physical blocks."""

    return _build_vision_parcae("vision_parcae_small_patch16_224", **kwargs)


def vision_parcae_base_patch16_224(**kwargs) -> VisionParcae:
    """Vision Parcae B/16 — 768-dim, 12 heads, 2+2+2 physical blocks."""

    return _build_vision_parcae("vision_parcae_base_patch16_224", **kwargs)
