import math
from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray


class LoRA(eqx.Module):
    """
    Very simple implementation of a LoRA[1] layer.

    References:
        [1] Hu, et al., LoRA: Low-Rank Adaptation of Large Language Models (2021).
    """

    scaling: float = eqx.field(static=True)

    A: Float[Array, "rank in"]
    B: Float[Array, "out rank"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: PRNGKeyArray,
        rank: int = 8,
        alpha: int = 16,
        dtype=jnp.float32,
        **kwargs,
    ):
        lim = 1 / math.sqrt(in_features)
        self.A = jr.uniform(
            key,
            (rank, in_features),
            dtype=dtype,
            minval=-lim,
            maxval=lim,
        )
        self.B = jnp.zeros((out_features, rank), dtype=dtype)

        self.scaling = alpha / rank

    def __call__(self, x: Array):
        x = self.B @ (self.A @ x)

        return x * self.scaling


class LayerSharing(eqx.Module):
    """
    Layer Sharing wrapper responsible for repeating a Callable,
    while learning LoRA params. This is similar to what is done in MobileLLM [1]
    and Zamba2 [2]. It allows to use more FLOPs without having to store more params.

    Depending on the position of this wrapper (block-level or layer-level), given three
    callables A, B and C, you can produce those to kind of repetition patterns:
    - ABCABC,
    - AABBCC.

    Note:
        Repeating a layer twice requires it to produce outputs of the same shape as its inputs

    References:
        [1] Liu, et al., MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases (2024).
        [2] https://github.com/Zyphra/Zamba2
    """

    repeat: int = eqx.field(static=True)

    loras: List[LoRA]
    dropouts: List[eqx.nn.Dropout]
    f: eqx.Module

    def __init__(
        self,
        dim: int,
        f: eqx.Module,
        repeat: int,
        *,
        key: PRNGKeyArray,
        rank: int = 8,
        alpha: int = 16,
        drop_rate: float = 0.0,
        **kwargs,
    ):
        assert repeat > 0

        keys = jr.split(key, repeat)
        self.repeat = repeat

        self.dropouts = [eqx.nn.Dropout(drop_rate) for i in range(self.repeat)]
        self.loras = [
            LoRA(
                in_features=dim,
                out_features=dim,
                rank=rank,
                alpha=alpha,
                key=keys[i],
            )
            for i in range(self.repeat)
        ]

        self.f = f

    def __call__(
        self,
        x: Array,
        *args,
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ):
        if self.repeat == 1:
            return self.f(
                x,
                *args,
                inference=inference,
                key=key,
                **kwargs,
            )

        keys = jr.split(key, self.repeat)
        reshape = len(x.shape) == 3

        for i in range(self.repeat):
            if reshape:
                _, h, w = x.shape
                lora_x = rearrange(x, "c h w -> (h w) c")
            else:
                lora_x = x
            lora_output = self.dropouts[i](
                jax.vmap(self.loras[i])(lora_x),
                inference=inference,
                key=keys[i],
            )
            if reshape:
                lora_output = rearrange(
                    lora_output,
                    "(h w) c -> c h w",
                    h=h,
                    w=w,
                )

            x = (
                self.f(
                    x,
                    *args,
                    inference=inference,
                    key=key,
                    **kwargs,
                )
                + lora_output
            )

        return x
