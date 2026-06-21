"""Delta compatibility tests for non-LoRA PEFT methods."""

from __future__ import annotations

import pytest
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft

from fixtures import TinyVisionTransformer


def test_prompt_delta_roundtrip(tmp_path, tiny_vision_transformer):
    model = eqft.apply_prompts(
        tiny_vision_transformer,
        eqft.PromptConfig(num_tokens=2),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "prompt.eqft"

    eqft.save_delta(model, path, method="prompt")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert jnp.array_equal(loaded.prompts[0], model.prompts[0])
    assert isinstance(loaded.config, eqft.PromptConfig)
    assert loaded.config.num_tokens == 2


def test_prefix_delta_roundtrip(tmp_path, tiny_vision_transformer):
    model = eqft.apply_prefixes(
        tiny_vision_transformer,
        eqft.PrefixConfig(num_prefix_tokens=2),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "prefix.eqft"

    eqft.save_delta(model, path, method="prefix")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert jnp.array_equal(loaded.prefixes[0], model.prefixes[0])
    assert isinstance(loaded.config, eqft.PrefixConfig)
    assert loaded.config.num_prefix_tokens == 2


def test_scale_shift_delta_roundtrip(tmp_path, tiny_vision_transformer):
    model = eqft.apply_scale_shift(
        tiny_vision_transformer,
        eqft.ScaleShiftConfig(target=eqft.TargetSpec(include=("*.norm",))),
    )
    wrapper = model.norm
    scale_shift = eqx.tree_at(
        lambda m: m.shift,
        wrapper.scale_shift,
        jnp.ones_like(wrapper.scale_shift.shift) * 0.25,
    )
    model = eqx.tree_at(lambda m: m.norm.scale_shift, model, scale_shift)
    path = tmp_path / "scale_shift.eqft"

    eqft.save_delta(model, path, method="scale_shift")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert jnp.array_equal(loaded.norm.scale_shift.shift, scale_shift.shift)


def test_ia3_delta_roundtrip(tmp_path, tiny_vision_transformer):
    model = eqft.apply_ia3(
        tiny_vision_transformer,
        eqft.IA3Config(target=eqft.TargetSpec(tags=("attention.proj",))),
    )
    ia3 = jnp.arange(4, dtype=jnp.float32) + 1.0
    model = eqx.tree_at(lambda m: m.blocks[0].attn.proj.ia3, model, ia3)
    path = tmp_path / "ia3.eqft"

    eqft.save_delta(model, path, method="ia3")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert jnp.array_equal(loaded.blocks[0].attn.proj.ia3, ia3)


def test_dora_delta_incompatible_architecture_raises(tmp_path):
    model = TinyVisionTransformer(key=jr.PRNGKey(0))
    dora = eqft.apply_dora(
        model,
        eqft.DoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    incompatible = TinyVisionTransformer(dim=5, hidden_dim=10, key=jr.PRNGKey(1))
    path = tmp_path / "dora.eqft"
    eqft.save_delta(dora, path, method="dora")

    with pytest.raises(ValueError, match="DoRA delta architecture hash mismatch"):
        eqft.load_delta(incompatible, path)
