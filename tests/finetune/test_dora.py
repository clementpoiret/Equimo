"""DoRA tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

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
            target=eqft.TargetSpec(tags=("attention.proj",)),
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


def test_dora_merge_removes_wrappers_and_preserves_outputs(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_dora(
        tiny_vision_transformer,
        eqft.DoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )

    merged = eqft.merge_dora(model)

    assert not isinstance(merged.blocks[0].attn.proj, eqft.DoRALinear)
    assert jnp.allclose(model(x), merged(x), atol=1e-6)
