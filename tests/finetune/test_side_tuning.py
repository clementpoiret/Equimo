"""Side-tuning tests."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


class IdentitySide(eqx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class NamedTapBackbone(eqx.Module):
    scale: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.scale * x

    def features_with_taps(self, x: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        return x, {
            "early": x + 1.0,
            "middle": x + 2.0,
            "late": x + 4.0,
        }


class SequenceTapBackbone(eqx.Module):
    scale: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.scale * x

    def forward_features_with_taps(
        self,
        x: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        return x, (x + 1.0, x + 2.0, x + 3.0, x + 4.0)


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


def test_lst_uses_named_activation_taps():
    model = eqft.SideTunedModel(
        NamedTapBackbone(jnp.asarray(10.0, dtype=jnp.float32)),
        IdentitySide(),
        eqft.LadderConnection(gate_init=1.0),
        eqft.LSTConfig(
            stop_gradient_backbone=False,
            tap_layers=("early", "late"),
        ),
    )
    x = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

    assert jnp.allclose(model(x), jnp.asarray([13.5, 24.5, 35.5], dtype=jnp.float32))


def test_lst_uses_percentage_activation_taps():
    model = eqft.SideTunedModel(
        SequenceTapBackbone(jnp.asarray(10.0, dtype=jnp.float32)),
        IdentitySide(),
        eqft.LadderConnection(gate_init=1.0),
        eqft.LSTConfig(
            stop_gradient_backbone=False,
            tap_layers=("25%", "100%"),
        ),
    )
    x = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

    assert jnp.allclose(model(x), jnp.asarray([13.5, 24.5, 35.5], dtype=jnp.float32))


def test_lst_stops_gradients_through_tap_activations():
    model = eqft.SideTunedModel(
        NamedTapBackbone(jnp.asarray(2.0, dtype=jnp.float32)),
        IdentitySide(),
        eqft.LadderConnection(gate_init=1.0),
        eqft.LSTConfig(tap_layers=("early", "late")),
    )
    x = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

    def loss_fn(model):
        return jnp.sum(model(x))

    _, grads = eqx.filter_value_and_grad(loss_fn)(model)

    assert jnp.array_equal(grads.backbone.scale, jnp.asarray(0.0, dtype=jnp.float32))


def test_apply_side_tuning_config_and_trainable_labels(tiny_linear_mlp):
    model = eqft.apply_side_tuning(
        tiny_linear_mlp,
        in_features=3,
        key=jr.PRNGKey(0),
        config=eqft.LSTConfig(
            tap_layers=("50%", "100%"),
            side_width_multiplier=0.5,
            gate_init=0.25,
        ),
    )
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="side_tuning",
            train_head=False,
        ),
    )

    assert model.config.stop_gradient_backbone
    assert model.side.head.layers[0].out_features == 2
    assert model.ladder.gate == jnp.asarray(0.25, dtype=jnp.float32)
    assert "side_tuning_decay" in plan.report.trainable_by_label
    assert plan.trainable.backbone.fc1.weight is None
    assert plan.trainable.side.head.layers[0].weight is not None
