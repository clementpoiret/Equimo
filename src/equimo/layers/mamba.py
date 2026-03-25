import math
from typing import Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.norm import RMSNormGated, get_norm
from equimo.ops.scan import non_causal_linear_attn

_MIXER_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_mixer(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to dynamically register new mixer modules.

    Why collision checking: Prevents third-party extensions from silently
    overwriting core layers, which can silently corrupt the computational graph.

    Args:
        name: Registry key. Defaults to the lowercase class name.
        force: If True, allow overwriting an existing entry. Default False.
    """

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _MIXER_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_MIXER_REGISTRY[registry_name]}."
            )

        _MIXER_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_mixer(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Get a mixer `eqx.Module` class from its registered name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _MIXER_REGISTRY:
        raise ValueError(
            f"Got an unknown module string: '{module}'. "
            f"Available modules: {list(_MIXER_REGISTRY.keys())}"
        )

    return _MIXER_REGISTRY[module_lower]


@register_mixer()
class Mamba2Mixer(eqx.Module):
    """Mamba2 Mixer.

    This class implements the Mamba2 Mixer using State Space Duality (SSD),
    from Mamba2 [1]. Also supports implementation details from Visual State Space
    Duality (VSSD) [2].

    Attributes:
        d_inner: Expanded inner dimension (dim * expand).
        n_heads: Number of SSM heads.
        head_dim: Dimension per SSM head.
        n_groups: Number of groups for grouped B/C projections.
        indices_xBC: Split indices for the xBC tensor.
        in_proj: Input projection layer.
        conv: Depthwise 1D convolution over the xBC channels.
        dt_bias: Bias for the softplus time step.
        A_log: Log of the SSM state-decay matrix A.
        D: Direct skip connection per head.
        norm: Post-SSM normalization layer.
        out_proj: Output projection layer.

    Args:
        dim: Input/output token dimension.
        key: PRNG key for parameter initialisation.
        expand: Channel expansion ratio for the inner dimension.
        n_groups: Number of groups for B/C projections.
        head_dim: Dimension per SSM head.
        d_state: SSM state size.
        d_conv: Depthwise convolution kernel size.
        dt_min: Minimum time-step for initialisation.
        dt_max: Maximum time-step for initialisation.
        dt_init_floor: Floor for the time-step initialisation.
        A_init_range: Uniform range for A initialisation.
        use_bias: Whether to use bias in linear projections.
        conv_bias: Whether to use bias in the depthwise convolution.
        norm_layer: Normalisation class applied after the SSM. Must accept
            ``dim`` as its sole positional argument. Use ``RMSNormGated``
            for fused gating, or any standard norm (e.g. ``eqx.nn.LayerNorm``)
            for additive gating via SiLU(z).

    Notes:
        This implementation is heavily based on wlln/scratch.

    References:
        [1] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
            (https://arxiv.org/abs/2401.04054)
        [2] VSSD: Vision Mamba with Non-Causal State Space Duality
            (https://arxiv.org/abs/2407.18559)
    """

    d_inner: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    n_groups: int = eqx.field(static=True)
    indices_xBC: List[int] = eqx.field(static=True)

    in_proj: eqx.nn.Linear
    conv: eqx.nn.Conv
    dt_bias: Float[Array, "n_heads"]
    A_log: Float[Array, "n_heads"]
    D: Float[Array, "n_heads"]
    norm: eqx.Module
    out_proj: eqx.nn.Linear

    def __init__(
        self,
        dim: int,
        *,
        key: PRNGKeyArray,
        expand: int = 2,
        n_groups: int = 1,
        head_dim: int = 64,
        d_state: int = 128,
        d_conv: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_init_range: Tuple[int, int] = (1, 16),
        use_bias: bool = False,
        conv_bias: bool = True,
        norm_layer: str | type[eqx.Module] = "layernorm",
        **kwargs,
    ):
        key_inproj, key_outproj, key_conv, key_randvals, key_a = jr.split(key, 5)
        norm_layer = get_norm(norm_layer)
        self.d_inner = int(dim * expand)
        if self.d_inner % head_dim != 0:
            raise ValueError("`d_inner` must be a multiple of `head_dim`.")
        self.n_heads = self.d_inner // head_dim
        self.head_dim = head_dim
        self.n_groups = n_groups
        self.indices_xBC = [self.d_inner, self.d_inner + n_groups * d_state]

        d_in_proj = 2 * self.d_inner + 2 * n_groups * d_state + self.n_heads
        self.in_proj = eqx.nn.Linear(
            dim,
            d_in_proj,
            use_bias=use_bias,
            key=key_inproj,
        )

        conv_dim = self.d_inner + 2 * n_groups * d_state
        self.conv = eqx.nn.Conv(
            num_spatial_dims=1,
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding="SAME",
            use_bias=conv_bias,
            key=key_conv,
        )

        rand_vals = jr.uniform(key_randvals, (self.n_heads,))
        dt = jnp.exp(
            rand_vals * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = jnp.clip(dt, min=dt_init_floor)
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))

        A_min, A_max = A_init_range
        A = jr.uniform(key_a, (self.n_heads,), minval=A_min, maxval=A_max)
        self.A_log = jnp.log(A)

        self.D = jnp.ones(self.n_heads)

        self.norm = norm_layer(self.d_inner)
        self.out_proj = eqx.nn.Linear(
            self.d_inner, dim, use_bias=use_bias, key=key_outproj
        )

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        *,
        key: PRNGKeyArray,
        inference: bool = False,
    ) -> Float[Array, "seqlen dim"]:
        dtype = x.dtype

        # Promote to float32 for numerical stability in exp/softplus
        A = -jnp.exp(self.A_log.astype(jnp.float32))

        zxbcdt = jax.vmap(self.in_proj)(x)

        z, xbc, dt = jnp.split(
            zxbcdt,
            [self.d_inner, zxbcdt.shape[-1] - self.n_heads],
            axis=-1,
        )

        dt = jax.nn.softplus(dt.astype(jnp.float32) + self.dt_bias.astype(jnp.float32))

        # apply 1d depthwise convolution and silu activation
        xbc_conv = rearrange(self.conv(rearrange(xbc, "s d -> d s")), "d s -> s d")
        xbc_silu = jax.nn.silu(xbc_conv[: x.shape[0], :])

        # split the conv output into the SSM inputs
        x_inner, B, C = jnp.split(xbc_silu, self.indices_xBC, axis=-1)

        x_inner = rearrange(x_inner, "l (h p) -> l h p", p=self.head_dim)

        y = non_causal_linear_attn(
            x_inner,
            dt=dt.astype(dtype),
            A=A.astype(dtype),
            B=B,
            C=C,
            D=self.D,
            n_groups=self.n_groups,
        )

        y = rearrange(y, "l h p -> l (h p)")

        # apply normalisation with SiLU gating
        if isinstance(self.norm, RMSNormGated):
            y = self.norm(y, jax.nn.silu(z))
        else:
            y = jax.vmap(self.norm)(y) * jax.nn.silu(z)

        y = jax.vmap(self.out_proj)(y)

        return y
