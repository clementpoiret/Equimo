"""PiSSA initialization tests."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


def test_pissa_initialization_shapes(tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.PiSSAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    module = model.blocks[0].attn.proj

    assert module.lora_A.shape == (2, 4)
    assert module.lora_B.shape == (4, 2)


def test_pissa_freeze_residual_preserves_base_outputs(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.PiSSAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)


def test_pissa_loraconfig_init_preserves_base_outputs(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            init="pissa",
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    module = model.blocks[0].attn.proj
    metadata = dict(module.metadata)

    assert module.base_weight_delta is not None
    assert metadata["method"] == "pissa"
    assert metadata["svd_effective"] == "exact_dense_top_rank"
    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)


def test_pissa_delta_roundtrip_preserves_residual_base(
    tmp_path,
    tiny_vision_transformer,
):
    x = jnp.ones((2, 3))
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.PiSSAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "pissa.eqft"

    eqft.save_delta(model, path, method="lora")
    bundle = eqft.extract_lora_delta(model)
    entry = bundle.adapter_config["entries"][0]
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert loaded.blocks[0].attn.proj.base_weight_delta is not None
    assert dict(entry["metadata"])["method"] == "pissa"
    assert jnp.allclose(
        entry["base_weight_delta"],
        -model.blocks[0].attn.proj.delta_weight(),
        atol=1e-6,
    )
    assert jnp.allclose(model(x), loaded(x), atol=1e-6)


def test_pissa_residual_delta_is_never_trainable(tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.PiSSAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="full"),
    )

    assert model.blocks[0].attn.proj.base_weight_delta is not None
    assert plan.trainable.blocks[0].attn.proj.base_weight_delta is None
    assert plan.frozen.blocks[0].attn.proj.base_weight_delta is not None


def test_pissa_config_exposes_spec_metadata():
    config = eqft.PiSSAConfig()

    assert config.residual_handling == "freeze_residual"
    assert config.fallback_init == "kaiming_A_zero_B"


def test_pissa_profile_declares_exact_svd_contract():
    profile = eqft.pissa_meng2024_profile()
    default_config_profile = eqft.pissa_meng2024_profile(eqft.PiSSAConfig())

    assert profile.id == "pissa.meng2024.exact_svd"
    assert profile.fidelity == "reference_implementation"
    assert profile.config["svd"] == "full"
    assert profile.config["niter"] == 0
    assert profile.required_artifacts == ("principal_svd_or_product",)
    assert default_config_profile.fidelity == "experimental"
    assert "randomized/iterative" in default_config_profile.known_deviations[0]
