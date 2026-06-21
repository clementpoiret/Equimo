"""VeRA tests."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


def test_vera_reproducible_frozen_random_bases(tiny_vision_transformer):
    config = eqft.VeRAConfig(
        rank=3,
        target=eqft.TargetSpec(tags=("attention.proj",)),
    )
    first = eqft.apply_vera(tiny_vision_transformer, config, key=jr.PRNGKey(0))
    second = eqft.apply_vera(tiny_vision_transformer, config, key=jr.PRNGKey(0))

    assert jnp.array_equal(first.blocks[0].attn.proj.vera_A, second.blocks[0].attn.proj.vera_A)
    assert jnp.array_equal(first.blocks[0].attn.proj.vera_B, second.blocks[0].attn.proj.vera_B)


def test_vera_zero_output_scale_identity(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_vera(
        tiny_vision_transformer,
        eqft.VeRAConfig(rank=3, target=eqft.TargetSpec(tags=("attention.proj",))),
        key=jr.PRNGKey(0),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
