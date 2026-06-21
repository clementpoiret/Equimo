"""Side-tuning tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


def test_lst_does_not_backprop_through_backbone(tiny_linear_mlp):
    model = eqft.SideTunedModel(
        tiny_linear_mlp,
        eqft.SideNetwork(3, 3, key=jr.PRNGKey(0)),
        eqft.LadderConnection(gate_init=1.0),
    )
    x = jnp.ones((4,))

    def loss_fn(model):
        return jnp.sum(model(x))

    _, grads = eqx.filter_value_and_grad(loss_fn)(model)

    assert jnp.array_equal(grads.backbone.fc1.weight, jnp.zeros_like(grads.backbone.fc1.weight))
    assert grads.side.head.layers[0].weight is not None
