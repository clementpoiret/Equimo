"""LoRA fine-tuning tests."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


class TinyExternalLinear(eqx.Module):
    """Linear-like stand-in for externally-owned quantized linear modules."""

    weight: jax.Array
    bias: jax.Array

    def __init__(self):
        self.weight = jnp.eye(3, dtype=jnp.float32)
        self.bias = jnp.zeros((3,), dtype=jnp.float32)

    def __call__(self, x):
        return self.weight @ x + self.bias


class TinyExternalLinearModel(eqx.Module):
    qlinear: TinyExternalLinear

    def __init__(self):
        self.qlinear = TinyExternalLinear()

    def __call__(self, x):
        return self.qlinear(x)


class TinyFanInFanOutLinear(eqx.Module):
    """Linear-like module storing weights as (in_features, out_features)."""

    weight: jax.Array
    bias: jax.Array

    def __init__(self):
        self.weight = jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10.0
        self.bias = jnp.array([0.5, -0.25], dtype=jnp.float32)

    def __call__(self, x):
        return x @ self.weight + self.bias


class TinyFanInFanOutLinearModel(eqx.Module):
    qlinear: TinyFanInFanOutLinear

    def __init__(self):
        self.qlinear = TinyFanInFanOutLinear()

    def __call__(self, x):
        return self.qlinear(x)


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


def test_lora_train_base_includes_wrapped_base_leaves(tiny_vision_transformer):
    lora = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            train_base=True,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        lora,
        trainable=eqft.TrainableSpec(mode="peft", method_name="lora"),
    )

    assert plan.trainable.blocks[0].attn.proj.lora_A is not None
    assert plan.trainable.blocks[0].attn.proj.lora_B is not None
    assert plan.trainable.blocks[0].attn.proj.base.weight is not None
    assert plan.trainable.blocks[0].attn.proj.base.bias is not None


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


def test_lora_accepts_external_linear_like_targets():
    model = TinyExternalLinearModel()
    lora = eqft.apply_lora(
        model,
        eqft.LoRAConfig(
            rank=1,
            alpha=1.0,
            target=eqft.TargetSpec(include=("qlinear",)),
        ),
        key=jr.PRNGKey(0),
    )

    assert isinstance(lora.qlinear, eqft.LoRALinear)
    assert isinstance(lora.qlinear.base, TinyExternalLinear)
    assert jnp.allclose(model(jnp.ones((3,))), lora(jnp.ones((3,))), atol=1e-6)


def test_lora_fan_in_fan_out_external_linear_merges_in_base_layout():
    model = TinyFanInFanOutLinearModel()
    lora = eqft.apply_lora(
        model,
        eqft.LoRAConfig(
            rank=1,
            alpha=2.0,
            fan_in_fan_out=True,
            target=eqft.TargetSpec(include=("qlinear",)),
        ),
        key=jr.PRNGKey(0),
    )

    assert isinstance(lora.qlinear, eqft.LoRALinear)
    assert lora.qlinear.lora_A.shape == (1, 3)
    assert lora.qlinear.lora_B.shape == (2, 1)
    assert lora.qlinear.delta_weight().shape == model.qlinear.weight.shape

    x = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
    assert jnp.allclose(model(x), lora(x), atol=1e-6)

    trained = eqx.tree_at(
        lambda tree: tree.qlinear.lora_B,
        lora,
        jnp.array([[0.25], [-0.5]], dtype=jnp.float32),
    )
    merged = eqft.merge_lora(trained)

    assert jnp.allclose(trained(x), merged(x), atol=1e-6)
    assert merged.qlinear.base.weight.shape == model.qlinear.weight.shape
