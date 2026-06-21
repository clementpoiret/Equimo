"""LoRA fine-tuning tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


def test_lora_zero_init_identity(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    lora = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(rank=2, alpha=4.0),
        key=jr.PRNGKey(0),
    )

    assert jnp.allclose(tiny_vision_transformer(x), lora(x), atol=1e-6)


def test_lora_trainable_only_lora_and_head(tiny_vision_transformer):
    lora = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(rank=2, alpha=4.0),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        lora,
        trainable=eqft.TrainableSpec(mode="peft", method_name="lora", train_head=True),
    )

    assert plan.trainable.blocks[0].attn.qkv.lora_A is not None
    assert plan.trainable.blocks[0].attn.qkv.lora_B is not None
    assert plan.trainable.blocks[0].attn.qkv.base.weight is None
    assert plan.trainable.head.weight is not None
    assert set(plan.report.trainable_by_label) == {
        "head_decay",
        "head_no_decay",
        "lora_A_decay",
        "lora_B_decay",
    }


def test_lora_filter_value_and_grad(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    lora = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(rank=2, alpha=4.0),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        lora,
        trainable=eqft.TrainableSpec(mode="peft", method_name="lora"),
    )

    def loss_fn(trainable):
        model = plan.combine(trainable)
        return jnp.sum(model(x))

    value, grads = eqx.filter_value_and_grad(loss_fn)(plan.trainable)

    assert value.shape == ()
    assert grads.blocks[0].attn.qkv.lora_B is not None
    assert grads.blocks[0].attn.qkv.base.weight is None


def test_lora_jit_compatible(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    lora = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(rank=2, alpha=4.0),
        key=jr.PRNGKey(0),
    )

    y = eqx.filter_jit(lora)(x)

    assert y.shape == (2,)


def test_lora_fused_qkv_shape(tiny_vision_transformer):
    lora = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags=("attention.qkv",)),
        ),
        key=jr.PRNGKey(0),
    )

    assert isinstance(lora.blocks[0].attn.qkv, eqft.LoRAMergedLinear)
    assert lora.blocks[0].attn.qkv.lora_B.shape == (
        tiny_vision_transformer.blocks[0].attn.qkv.weight.shape[0],
        2,
    )
