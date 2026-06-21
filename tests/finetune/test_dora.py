"""DoRA tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft


def test_dora_zero_update_preserves_base_outputs(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    dora = eqft.apply_dora(
        tiny_vision_transformer,
        eqft.DoRAConfig(rank=2, alpha=4.0),
        key=jr.PRNGKey(0),
    )

    assert jnp.allclose(tiny_vision_transformer(x), dora(x), atol=1e-5)


def test_dora_gradients_flow_to_magnitude_and_low_rank(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    dora = eqft.apply_dora(
        tiny_vision_transformer,
        eqft.DoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        dora,
        trainable=eqft.TrainableSpec(mode="peft", method_name="dora"),
    )

    def loss_fn(trainable):
        model = plan.combine(trainable)
        return jnp.sum(model(x))

    _, grads = eqx.filter_value_and_grad(loss_fn)(plan.trainable)

    assert grads.blocks[0].attn.proj.magnitude is not None
    assert grads.blocks[0].attn.proj.lora_A is not None
    assert grads.blocks[0].attn.proj.lora_B is not None
    assert grads.blocks[0].attn.proj.base.weight is None


def test_dora_train_base_includes_wrapped_base_leaves(tiny_vision_transformer):
    dora = eqft.apply_dora(
        tiny_vision_transformer,
        eqft.DoRAConfig(
            rank=2,
            alpha=4.0,
            train_base=True,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        dora,
        trainable=eqft.TrainableSpec(mode="peft", method_name="dora"),
    )

    assert plan.trainable.blocks[0].attn.proj.magnitude is not None
    assert plan.trainable.blocks[0].attn.proj.lora_A is not None
    assert plan.trainable.blocks[0].attn.proj.lora_B is not None
    assert plan.trainable.blocks[0].attn.proj.base.weight is not None
    assert plan.trainable.blocks[0].attn.proj.base.bias is not None


def test_dora_magnitude_init_base_weight_norm(tiny_vision_transformer):
    dora = eqft.apply_dora(
        tiny_vision_transformer,
        eqft.DoRAConfig(
            rank=2,
            alpha=4.0,
            magnitude_init="base_weight_norm",
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )

    assert jnp.allclose(
        dora.blocks[0].attn.proj.magnitude,
        jnp.linalg.norm(tiny_vision_transformer.blocks[0].attn.proj.weight, axis=1),
    )


def test_dora_magnitude_init_rejects_unsupported_policy(tiny_vision_transformer):
    with pytest.raises(ValueError, match="magnitude_init"):
        eqft.apply_dora(
            tiny_vision_transformer,
            eqft.DoRAConfig(
                rank=2,
                alpha=4.0,
                magnitude_init="zeros",
                target=eqft.TargetSpec(tags_any=("attention.proj",)),
            ),
            key=jr.PRNGKey(0),
        )


def test_dora_dropout_requires_key_and_affects_low_rank_branch(tiny_vision_transformer):
    dora = eqft.apply_dora(
        tiny_vision_transformer,
        eqft.DoRAConfig(
            rank=2,
            alpha=4.0,
            dropout=0.5,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    module = dora.blocks[0].attn.proj
    module = eqx.tree_at(
        lambda m: m.lora_B,
        module,
        jnp.ones_like(module.lora_B) * 0.1,
    )

    with pytest.raises(ValueError, match="PRNG key"):
        module(jnp.ones((4,)), inference=False)

    first = module(jnp.ones((4,)), key=jr.PRNGKey(1), inference=False)
    second = module(jnp.ones((4,)), key=jr.PRNGKey(2), inference=False)

    assert not jnp.allclose(first, second)


def test_dora_merge_removes_wrappers_and_preserves_outputs(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_dora(
        tiny_vision_transformer,
        eqft.DoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )

    merged = eqft.merge_dora(model)

    assert not isinstance(merged.blocks[0].attn.proj, eqft.DoRALinear)
    assert jnp.allclose(model(x), merged(x), atol=1e-6)
