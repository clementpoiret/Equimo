"""VeRA tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


def test_vera_reproducible_frozen_random_bases(tiny_vision_transformer):
    config = eqft.VeRAConfig(
        rank=3,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
    )
    first = eqft.apply_vera(tiny_vision_transformer, config, key=jr.PRNGKey(0))
    second = eqft.apply_vera(tiny_vision_transformer, config, key=jr.PRNGKey(0))

    assert jnp.array_equal(
        first.blocks[0].attn.proj.vera_A, second.blocks[0].attn.proj.vera_A
    )
    assert jnp.array_equal(
        first.blocks[0].attn.proj.vera_B, second.blocks[0].attn.proj.vera_B
    )


def test_vera_shared_reuses_compatible_random_bases(tiny_vision_transformer):
    shared = eqft.apply_vera(
        tiny_vision_transformer,
        eqft.VeRAConfig(
            rank=3,
            shared=True,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    unshared = eqft.apply_vera(
        tiny_vision_transformer,
        eqft.VeRAConfig(
            rank=3,
            shared=False,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )

    assert jnp.array_equal(
        shared.blocks[0].attn.proj.vera_A,
        shared.blocks[1].attn.proj.vera_A,
    )
    assert jnp.array_equal(
        shared.blocks[0].attn.proj.vera_B,
        shared.blocks[1].attn.proj.vera_B,
    )
    assert shared.blocks[0].attn.proj.share_scope == "shape_compatible"
    assert (
        shared.blocks[0].attn.proj.basis_pool_key
        == shared.blocks[1].attn.proj.basis_pool_key
    )
    assert (
        shared.blocks[0].attn.proj.basis_key_data
        == shared.blocks[1].attn.proj.basis_key_data
    )
    assert not jnp.array_equal(
        unshared.blocks[0].attn.proj.vera_A,
        unshared.blocks[1].attn.proj.vera_A,
    )
    assert unshared.blocks[0].attn.proj.share_scope == "per_module"


def test_vera_zero_output_scale_identity(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_vera(
        tiny_vision_transformer,
        eqft.VeRAConfig(rank=3, target=eqft.TargetSpec(tags_any=("attention.proj",))),
        key=jr.PRNGKey(0),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)


def test_merge_vera_preserves_outputs_and_removes_wrapper(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_vera(
        tiny_vision_transformer,
        eqft.VeRAConfig(rank=3, target=eqft.TargetSpec(tags_any=("attention.proj",))),
        key=jr.PRNGKey(0),
    )
    model = eqx.tree_at(
        lambda tree: tree.blocks[0].attn.proj.vera_output_scale,
        model,
        jnp.asarray([0.5, -0.25, 0.75, 1.0]),
    )

    merged = eqft.merge_vera(model)

    assert not isinstance(merged.blocks[0].attn.proj, eqft.VeRALinear)
    assert jnp.allclose(model(x), merged(x), atol=1e-6)
