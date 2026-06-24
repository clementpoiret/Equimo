"""Prefix tuning tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft


def test_prefix_trainable_only_prefixes(tiny_vision_transformer):
    prefixed = eqft.apply_prefixes(
        tiny_vision_transformer,
        eqft.PrefixConfig(num_prefix_tokens=2),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        prefixed,
        trainable=eqft.TrainableSpec(
            mode="peft", method_name="prefix", train_head=False
        ),
    )

    assert plan.trainable.prefixes[0] is not None
    assert plan.trainable.prefix_projections[0].down.weight is not None
    assert plan.trainable.base.patch_embed.proj.weight is None
    assert "prefix_decay" in plan.report.trainable_by_label


def test_prefix_projection_builds_kv_state(tiny_vision_transformer):
    prefixed = eqft.apply_prefixes(
        tiny_vision_transformer,
        eqft.PrefixConfig(num_prefix_tokens=2, prefix_projection=True),
        key=jr.PRNGKey(0),
    )

    projection = prefixed.prefix_projections[0]
    assert isinstance(projection, eqft.PrefixProjection)
    assert jnp.array_equal(
        prefixed.base.blocks[0].attn.state,
        projection(prefixed.prefixes[0]),
    )


def test_prefix_projection_can_be_disabled(tiny_vision_transformer):
    prefixed = eqft.apply_prefixes(
        tiny_vision_transformer,
        eqft.PrefixConfig(num_prefix_tokens=2, prefix_projection=False),
        key=jr.PRNGKey(0),
    )

    assert prefixed.prefix_projections[0] is None


def test_prefix_direct_kv_is_explicitly_unsupported(tiny_vision_transformer):
    with pytest.raises(ValueError, match="direct_kv"):
        eqft.apply_prefixes(
            tiny_vision_transformer,
            eqft.PrefixConfig(num_prefix_tokens=2, direct_kv=True),
            key=jr.PRNGKey(0),
        )


def test_prefix_target_rejects_unsupported_target(tiny_vision_transformer):
    with pytest.raises(ValueError, match="target"):
        eqft.apply_prefixes(
            tiny_vision_transformer,
            eqft.PrefixConfig(num_prefix_tokens=2, target=("attention.q",)),
            key=jr.PRNGKey(0),
        )


def test_prefix_dropout_requires_key_when_training(tiny_vision_transformer):
    prefixed = eqft.apply_prefixes(
        tiny_vision_transformer,
        eqft.PrefixConfig(num_prefix_tokens=2, prefix_dropout=0.5),
        key=jr.PRNGKey(0),
    )
    attention = prefixed.base.blocks[0].attn

    assert attention.prefix_dropout == 0.5
    with pytest.raises(ValueError, match="prefix dropout"):
        attention(jnp.ones((2, 4)), inference=False)
    attention(jnp.ones((2, 4)), key=jr.PRNGKey(1), inference=False)


def test_prefix_attention_extends_key_value_mask(tiny_vision_transformer):
    prefixed = eqft.apply_prefixes(
        tiny_vision_transformer,
        eqft.PrefixConfig(num_prefix_tokens=2),
        key=jr.PRNGKey(0),
    )
    attention = prefixed.base.blocks[0].attn
    x = jnp.ones((2, 4), dtype=jnp.float32)
    mask = jnp.ones((1, 2, 2), dtype=jnp.int32)

    y = attention(x, mask=mask, inference=True)

    assert attention.state.shape[2] == 2
    assert y.shape == x.shape


def test_prefixes_receive_gradients(tiny_vision_transformer):
    prefixed = eqft.apply_prefixes(
        tiny_vision_transformer,
        eqft.PrefixConfig(num_prefix_tokens=2),
        key=jr.PRNGKey(0),
    )
    x = jnp.ones((2, 3))

    def loss_fn(prefixes):
        local = eqx.tree_at(lambda m: m.prefixes, prefixed, prefixes)
        return jnp.sum(local(x))

    grads = eqx.filter_grad(loss_fn)(prefixed.prefixes)

    assert any(jnp.any(grad != 0) for grad in grads)
