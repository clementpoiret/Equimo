"""Delta compatibility tests for non-LoRA PEFT methods."""

from __future__ import annotations

from dataclasses import replace

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


def test_soft_prompt_delta_preserves_config_subclass(tmp_path, tiny_text_encoder):
    model = eqft.apply_prompts(
        tiny_text_encoder,
        eqft.SoftPromptConfig(num_tokens=2),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "soft_prompt.eqft"

    eqft.save_delta(model, path, method="prompt")
    loaded = eqft.load_delta(tiny_text_encoder, path)

    assert isinstance(loaded.config, eqft.SoftPromptConfig)
    assert loaded.config.prepend_to == "input"
    assert jnp.array_equal(loaded.prompts[0], model.prompts[0])


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
    assert isinstance(loaded.prefix_projections[0], eqft.PrefixProjection)
    assert jnp.array_equal(
        loaded.prefix_projections[0].down.weight,
        model.prefix_projections[0].down.weight,
    )
    assert isinstance(loaded.config, eqft.PrefixConfig)
    assert loaded.config.num_prefix_tokens == 2
    assert jnp.allclose(
        loaded(jnp.ones((2, 3))),
        model(jnp.ones((2, 3))),
        atol=1e-6,
    )


def test_scale_shift_delta_roundtrip(tmp_path, tiny_vision_transformer):
    model = eqft.apply_scale_shift(
        tiny_vision_transformer,
        eqft.ScaleShiftConfig(
            target=eqft.TargetSpec(include=("*.norm",)),
            mergeable=False,
        ),
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
    assert loaded.norm.mergeable is False


def test_scale_shift_delta_incompatible_shape_raises_bundle_error(tmp_path, tiny_vision_transformer):
    model = eqft.apply_scale_shift(
        tiny_vision_transformer,
        eqft.ScaleShiftConfig(target=eqft.TargetSpec(include=("*.norm",))),
    )
    path = tmp_path / "scale_shift.eqft"
    bundle = eqft.save_delta(model, path, method="scale_shift")
    entries = [dict(entry) for entry in bundle.adapter_config["entries"]]
    entries[0]["scale"] = jnp.ones((5,), dtype=jnp.float32)
    entries[0]["shift"] = jnp.zeros((5,), dtype=jnp.float32)
    bad_bundle = replace(bundle, adapter_config={"entries": entries})

    with pytest.raises(eqft.FineTuneBundleError, match="feature dimension"):
        eqft.load_delta(tiny_vision_transformer, bad_bundle)


def test_ia3_delta_roundtrip(tmp_path, tiny_vision_transformer):
    model = eqft.apply_ia3(
        tiny_vision_transformer,
        eqft.IA3Config(
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
            mergeable=False,
        ),
    )
    ia3 = jnp.arange(4, dtype=jnp.float32) + 1.0
    model = eqx.tree_at(lambda m: m.blocks[0].attn.proj.ia3, model, ia3)
    path = tmp_path / "ia3.eqft"

    eqft.save_delta(model, path, method="ia3")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert jnp.array_equal(loaded.blocks[0].attn.proj.ia3, ia3)
    assert loaded.blocks[0].attn.proj.mergeable is False


def test_ia3_delta_missing_path_raises_bundle_error(tmp_path, tiny_vision_transformer):
    model = eqft.apply_ia3(
        tiny_vision_transformer,
        eqft.IA3Config(target=eqft.TargetSpec(tags_any=("attention.proj",))),
    )
    path = tmp_path / "ia3.eqft"
    bundle = eqft.save_delta(model, path, method="ia3")
    entries = [dict(entry) for entry in bundle.adapter_config["entries"]]
    entries[0]["path"] = "blocks.99.attn.proj"
    bad_bundle = replace(bundle, adapter_config={"entries": entries})

    with pytest.raises(eqft.FineTuneBundleError, match="no matching leaf"):
        eqft.load_delta(tiny_vision_transformer, bad_bundle)


def test_vera_delta_roundtrip(tmp_path, tiny_vision_transformer):
    model = eqft.apply_vera(
        tiny_vision_transformer,
        eqft.VeRAConfig(
            rank=3,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    output_scale = jnp.arange(4, dtype=jnp.float32) * 0.1
    model = eqx.tree_at(
        lambda m: m.blocks[0].attn.proj.vera_output_scale,
        model,
        output_scale,
    )
    path = tmp_path / "vera.eqft"

    bundle = eqft.save_delta(model, path, method="vera")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert bundle.method == "vera"
    assert bundle.metadata["trainable_params"] == 14
    assert jnp.array_equal(loaded.blocks[0].attn.proj.vera_A, model.blocks[0].attn.proj.vera_A)
    assert jnp.array_equal(loaded.blocks[0].attn.proj.vera_B, model.blocks[0].attn.proj.vera_B)
    assert jnp.array_equal(loaded.blocks[0].attn.proj.vera_output_scale, output_scale)
    assert jnp.allclose(
        loaded(jnp.ones((2, 3))),
        model(jnp.ones((2, 3))),
        atol=1e-6,
    )


def test_side_tuning_delta_roundtrip(tmp_path, tiny_linear_mlp):
    model = eqft.apply_side_tuning(
        tiny_linear_mlp,
        in_features=3,
        key=jr.PRNGKey(0),
        config=eqft.LSTConfig(
            tap_layers=("50%", "100%"),
            side_width_multiplier=0.5,
            gate_init=0.5,
        ),
    )
    path = tmp_path / "side_tuning.eqft"
    x = jnp.ones((4,))

    bundle = eqft.save_delta(model, path, method="side_tuning")
    loaded = eqft.load_delta(tiny_linear_mlp, path)

    assert bundle.method == "side_tuning"
    assert isinstance(loaded, eqft.SideTunedModel)
    assert loaded.config.tap_layers == ("50%", "100%")
    assert jnp.array_equal(loaded.ladder.gate, model.ladder.gate)
    assert jnp.allclose(loaded(x), model(x), atol=1e-6)


def test_dora_delta_incompatible_architecture_raises(tmp_path):
    model = TinyVisionTransformer(key=jr.PRNGKey(0))
    dora = eqft.apply_dora(
        model,
        eqft.DoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    incompatible = TinyVisionTransformer(dim=5, hidden_dim=10, key=jr.PRNGKey(1))
    path = tmp_path / "dora.eqft"
    eqft.save_delta(dora, path, method="dora")

    with pytest.raises(eqft.FineTuneBundleError, match="DoRA delta architecture hash mismatch"):
        eqft.load_delta(incompatible, path)
