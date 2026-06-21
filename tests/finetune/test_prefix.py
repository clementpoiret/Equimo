"""Prefix tuning tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


def test_prefix_trainable_only_prefixes(tiny_vision_transformer):
    prefixed = eqft.apply_prefixes(
        tiny_vision_transformer,
        eqft.PrefixConfig(num_prefix_tokens=2),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        prefixed,
        trainable=eqft.TrainableSpec(mode="peft", method_name="prefix", train_head=False),
    )

    assert plan.trainable.prefixes[0] is not None
    assert plan.trainable.base.patch_embed.proj.weight is None
    assert "prefix_decay" in plan.report.trainable_by_label


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
