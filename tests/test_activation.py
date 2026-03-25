"""Tests for equimo.layers.activation."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.activation import _ACT_REGISTRY, get_act, register_act

KEY = jr.PRNGKey(0)
X = jnp.ones((4, 8), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# get_act
# ---------------------------------------------------------------------------


class TestGetAct:
    def test_string_relu(self):
        fn = get_act("relu")
        assert fn is jax.nn.relu

    def test_string_gelu(self):
        fn = get_act("gelu")
        assert fn is jax.nn.gelu

    def test_string_silu(self):
        fn = get_act("silu")
        assert fn is jax.nn.silu

    def test_string_elu(self):
        fn = get_act("elu")
        assert fn is jax.nn.elu

    def test_string_sigmoid(self):
        fn = get_act("sigmoid")
        assert fn is jax.nn.sigmoid

    def test_string_hard_sigmoid(self):
        fn = get_act("hard_sigmoid")
        assert fn is jax.nn.hard_sigmoid

    def test_string_hard_swish(self):
        fn = get_act("hard_swish")
        assert fn is jax.nn.hard_swish

    def test_string_softmax(self):
        fn = get_act("softmax")
        assert fn is jax.nn.softmax

    def test_string_case_insensitive(self):
        assert get_act("ReLU") is get_act("relu")
        assert get_act("GELU") is get_act("gelu")

    def test_callable_passthrough(self):
        fn = lambda x: x
        assert get_act(fn) is fn

    def test_jax_callable_passthrough(self):
        assert get_act(jax.nn.relu) is jax.nn.relu

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown activation string"):
            get_act("nonexistent_act")

    def test_exactgelu_registered(self):
        fn = get_act("exactgelu")
        out = fn(X)
        assert out.shape == X.shape
        assert jnp.all(jnp.isfinite(out))

    def test_all_builtins_callable(self):
        builtins = ["relu", "gelu", "exactgelu", "silu", "elu", "sigmoid",
                    "hard_sigmoid", "hard_swish", "softmax"]
        for name in builtins:
            fn = get_act(name)
            out = fn(X)
            assert out.shape == X.shape, f"{name} changed shape"
            assert jnp.all(jnp.isfinite(out)), f"{name} produced non-finite output"


# ---------------------------------------------------------------------------
# register_act
# ---------------------------------------------------------------------------


class TestRegisterAct:
    def test_register_default_name(self):
        @register_act()
        def my_custom_act(x):
            return x * 2

        assert "my_custom_act" in _ACT_REGISTRY
        assert get_act("my_custom_act") is my_custom_act

    def test_register_custom_name(self):
        @register_act(name="MySwish")
        def another_act(x):
            return x * jax.nn.sigmoid(x)

        assert "myswish" in _ACT_REGISTRY
        assert get_act("myswish") is another_act

    def test_register_preserves_function(self):
        @register_act(name="identity_test")
        def identity(x):
            return x

        fn = get_act("identity_test")
        out = fn(X)
        assert jnp.array_equal(out, X)

    def test_register_duplicate_raises(self):
        @register_act(name="dup_act_1")
        def act_a(x):
            return x

        with pytest.raises(ValueError, match="already registered"):
            @register_act(name="dup_act_1")
            def act_b(x):
                return x

    def test_register_returns_original_function(self):
        def raw_fn(x):
            return x + 1

        decorated = register_act(name="reg_returns_test")(raw_fn)
        assert decorated is raw_fn
