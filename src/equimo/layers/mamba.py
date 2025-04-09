import math
from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.norm import RMSNormGated
from equimo.ops.scan import non_causal_linear_attn


class Mamba2Mixer(eqx.Module):
    """Mamba2 Mixer.

    This class implements the a Mamba2 Mixer using State Space Duality (SSD),
    from Mamba2 [1]. Also supports implementation details from Visual State Space
    Duality (VSSD) [2].

    Attributes:
        in_proj (eqx.nn.Linear): Input projection layer.
        conv (eqx.nn.Conv): Convolutional layer for processing input.
        dt_bias (eqx.nn.Param): Bias for delta time.
        A_log (eqx.nn.Param): Logarithm of the state transition matrix A.
        D (eqx.nn.Param): Direct feedthrough matrix D.
        norm (eqx.nn.RMSNorm): Root Mean Square Layer Normalization.
        out_proj (eqx.nn.Linear): Output projection layer.

    Args:
        config (MambaConfig): Configuration object for the Mamba2VisionMixer.
        rngs (nnx.Rngs): Random number generators for parameter initialization.

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
    norm: eqx.nn.LayerNorm
    out_proj: eqx.nn.Linear

    def __init__(
        self,
        d_model: int,
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
        **kwargs,
    ):
        key_inproj, key_outproj, key_conv, key_randvals, key_a = jr.split(key, 5)
        self.d_inner = int(d_model * expand)
        if self.d_inner % head_dim != 0:
            raise ValueError("`d_inner` must be a multiple of `head_dim`.")
        self.n_heads = self.d_inner // head_dim
        self.head_dim = head_dim
        self.n_groups = n_groups
        self.indices_xBC = [self.d_inner, self.d_inner + n_groups * d_state]

        d_in_proj = 2 * self.d_inner + 2 * n_groups * d_state + self.n_heads
        self.in_proj = eqx.nn.Linear(
            d_model,
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
        dt = jnp.clip(dt, a_min=dt_init_floor)
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))

        A_min, A_max = A_init_range
        A = jr.uniform(key_a, (self.n_heads,), minval=A_min, maxval=A_max)
        self.A_log = jnp.log(A)

        self.D = jnp.ones(self.n_heads)

        self.norm = eqx.nn.LayerNorm(self.d_inner)
        self.out_proj = eqx.nn.Linear(
            self.d_inner, d_model, use_bias=use_bias, key=key_outproj
        )

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        A = -jnp.exp(self.A_log)
        zxbcdt = jax.vmap(self.in_proj)(x)

        z, xbc, dt = jnp.split(
            zxbcdt,
            [self.d_inner, zxbcdt.shape[-1] - self.n_heads],
            axis=-1,
        )

        dt = jax.nn.softplus(dt + self.dt_bias)

        # Pad or truncate the xbc tensor to match the conv kernel size
        # xbc_rearranged = rearrange(xbc, "b l d -> b d l")
        # conv_state = pad_or_truncate_to_length(xbc_rearranged, self.config.d_conv)

        # apply 1d convolution and silu activation
        xbc_conv = rearrange(self.conv(rearrange(xbc, "s d -> d s")), "d s -> s d")
        xbc_silu = jax.nn.silu(xbc_conv[: x.shape[0], :])

        # split the conv state into the conv kernel and the conv state
        x, B, C = jnp.split(xbc_silu, self.indices_xBC, axis=-1)

        x = rearrange(x, "l (h p) -> l h p", p=self.head_dim)

        y = non_causal_linear_attn(
            x, dt=dt, A=A, B=B, C=C, D=self.D, n_groups=self.n_groups
        )

        y = rearrange(y, "l h p -> l (h p)")

        # apply the output projection
        if isinstance(self.norm, RMSNormGated):
            y = self.norm(y, jax.nn.silu(z))
        else:
            # Should be LayerNorm
            y = jax.vmap(self.norm)(y) * z

        y = jax.vmap(self.out_proj)(y)

        return y
