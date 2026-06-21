"""LoRA delta serialization tests."""

from __future__ import annotations

import pytest
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft

from fixtures import TinyVisionTransformer


def _trained_like_lora(model):
    lora = eqft.apply_lora(
        model,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    module = lora.blocks[0].attn.proj
    module = eqx.tree_at(
        lambda m: m.lora_B,
        module,
        jnp.ones_like(module.lora_B) * 0.01,
    )
    return eqx.tree_at(lambda m: m.blocks[0].attn.proj, lora, module)


def test_lora_delta_roundtrip(tmp_path):
    model = TinyVisionTransformer(key=jr.PRNGKey(0))
    lora = _trained_like_lora(model)
    x = jnp.ones((2, 3))
    path = tmp_path / "delta.eqft"

    eqft.save_delta(lora, path)
    loaded = eqft.load_delta(model, path)

    assert jnp.allclose(lora(x), loaded(x), atol=1e-6)


def test_lora_delta_incompatible_shape_raises(tmp_path):
    model = TinyVisionTransformer(key=jr.PRNGKey(0))
    lora = _trained_like_lora(model)
    incompatible = TinyVisionTransformer(dim=5, hidden_dim=10, key=jr.PRNGKey(1))
    path = tmp_path / "delta.eqft"
    eqft.save_delta(lora, path)

    with pytest.raises(eqft.FineTuneBundleError, match="architecture hash mismatch"):
        eqft.load_delta(incompatible, path)
