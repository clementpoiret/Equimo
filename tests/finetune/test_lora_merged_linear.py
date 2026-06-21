"""LoRA merge/unmerge tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


def _nonzero_lora(model):
    module = model.blocks[0].attn.proj
    module = eqx.tree_at(
        lambda m: m.lora_B,
        module,
        jnp.ones_like(module.lora_B) * 0.01,
    )
    return eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, module)


def test_lora_merge_equivalence(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    lora = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    lora = _nonzero_lora(lora)

    merged = eqft.merge_lora(lora)

    assert merged.blocks[0].attn.proj.merged is True
    assert jnp.allclose(lora(x), merged(x), atol=1e-6)


def test_lora_unmerge_roundtrip(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    lora = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    lora = _nonzero_lora(lora)

    unmerged = eqft.unmerge_lora(eqft.merge_lora(lora))

    assert unmerged.blocks[0].attn.proj.merged is False
    assert jnp.allclose(lora(x), unmerged(x), atol=1e-6)
