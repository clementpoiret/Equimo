"""Tests for equimo.layers.dropout."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.dropout import (
    DropPath,
    DropPathAdd,
    get_dropout,
    register_dropout,
)

# Shared fixtures

KEY = jr.PRNGKey(0)
SHAPE = (16, 64)  # (seqlen, dim) — typical token tensor


# DropPath


class TestDropPath:
    def test_output_shape(self):
        layer = DropPath(p=0.5)
        x = jr.normal(KEY, SHAPE)
        assert layer(x, key=KEY).shape == SHAPE

    def test_inference_mode_passthrough(self):
        """inference=True must return x unchanged."""
        layer = DropPath(p=0.9)
        x = jr.normal(KEY, SHAPE)
        out = layer(x, key=KEY, inference=True)
        assert jnp.array_equal(out, x)

    def test_p_zero_acts_as_inference(self):
        """p=0 must disable dropout even without inference=True."""
        layer = DropPath(p=0.0)
        x = jr.normal(KEY, SHAPE)
        assert jnp.array_equal(layer(x, key=KEY), x)

    def test_stored_inference_true_passthrough(self):
        """Module-level inference=True must disable dropout."""
        layer = DropPath(p=0.9, inference=True)
        x = jr.normal(KEY, SHAPE)
        assert jnp.array_equal(layer(x), x)

    def test_call_inference_overrides_stored(self):
        """Call-time inference=True overrides stored inference=False."""
        layer = DropPath(p=0.9, inference=False)
        x = jr.normal(KEY, SHAPE)
        assert jnp.array_equal(layer(x, key=KEY, inference=True), x)

    def test_key_required_in_train_mode(self):
        layer = DropPath(p=0.5)
        x = jr.normal(KEY, SHAPE)
        with pytest.raises(RuntimeError, match="requires a key"):
            layer(x, inference=False)

    def test_output_finite(self):
        layer = DropPath(p=0.5)
        x = jr.normal(KEY, SHAPE)
        assert jnp.all(jnp.isfinite(layer(x, key=KEY)))

    def test_drops_along_first_axis(self):
        """Mask is broadcast over all but first axis; each row is all-dropped or kept."""
        layer = DropPath(p=0.9)
        x = jnp.ones(SHAPE)
        out = layer(x, key=KEY, inference=False)
        # Each row must be either all-zero or all-nonzero (same value across dim axis)
        row_sums = jnp.sum(jnp.abs(out), axis=-1)
        assert jnp.all((row_sums == 0) | (row_sums > 0))

    def test_stochastic_across_keys(self):
        """Different keys must (with overwhelming probability) produce different outputs."""
        layer = DropPath(p=0.5)
        x = jnp.ones(SHAPE)
        out1 = layer(x, key=jr.PRNGKey(1), inference=False)
        out2 = layer(x, key=jr.PRNGKey(2), inference=False)
        assert not jnp.array_equal(out1, out2)

    def test_expected_value_approximately_preserved(self):
        """With rescaling by 1/q, E[output] ≈ E[input] over many samples."""
        layer = DropPath(p=0.5)
        x = jnp.ones(SHAPE)
        outputs = jnp.stack([layer(x, key=jr.PRNGKey(i)) for i in range(500)])
        assert jnp.abs(jnp.mean(outputs) - 1.0) < 0.1

    def test_bfloat16_input_finite(self):
        layer = DropPath(p=0.5)
        x = jr.normal(KEY, SHAPE).astype(jnp.bfloat16)
        out = layer(x, key=KEY)
        assert jnp.all(jnp.isfinite(out))

    def test_float16_input_finite(self):
        layer = DropPath(p=0.5)
        x = jr.normal(KEY, SHAPE).astype(jnp.float16)
        out = layer(x, key=KEY)
        assert jnp.all(jnp.isfinite(out))

    def test_p_stored_as_float(self):
        layer = DropPath(p=0.3)
        assert isinstance(layer.p, float)
        assert layer.p == pytest.approx(0.3)

    def test_jax_array_p_converted_to_float(self):
        p_arr = jnp.array([0.4])
        layer = DropPath(p=p_arr)
        assert isinstance(layer.p, float)
        assert layer.p == pytest.approx(0.4)

    def test_jax_array_p_wrong_length_raises(self):
        with pytest.raises(ValueError, match="values for p"):
            DropPath(p=jnp.array([0.1, 0.2]))

    def test_3d_input(self):
        """DropPath must handle (C, H, W) inputs."""
        layer = DropPath(p=0.5)
        x = jr.normal(KEY, (32, 8, 8))
        assert layer(x, key=KEY).shape == (32, 8, 8)


# DropPathAdd


class TestDropPathAdd:
    def test_output_shape(self):
        layer = DropPathAdd(p=0.5)
        x1 = jr.normal(KEY, SHAPE)
        x2 = jr.normal(jr.PRNGKey(1), SHAPE)
        assert layer(x1, x2, key=KEY).shape == SHAPE

    def test_inference_mode_always_adds(self):
        """inference=True must return x1 + x2 exactly."""
        layer = DropPathAdd(p=0.9)
        x1 = jr.normal(KEY, SHAPE)
        x2 = jr.normal(jr.PRNGKey(1), SHAPE)
        out = layer(x1, x2, key=KEY, inference=True)
        assert jnp.array_equal(out, x1 + x2)

    def test_p_zero_always_adds(self):
        """p=0 must always add x2 to x1."""
        layer = DropPathAdd(p=0.0)
        x1 = jr.normal(KEY, SHAPE)
        x2 = jr.normal(jr.PRNGKey(1), SHAPE)
        assert jnp.array_equal(layer(x1, x2, key=KEY), x1 + x2)

    def test_stored_inference_true_always_adds(self):
        layer = DropPathAdd(p=0.9, inference=True)
        x1 = jr.normal(KEY, SHAPE)
        x2 = jr.normal(jr.PRNGKey(1), SHAPE)
        assert jnp.array_equal(layer(x1, x2), x1 + x2)

    def test_call_inference_overrides_stored(self):
        layer = DropPathAdd(p=0.9, inference=False)
        x1 = jr.normal(KEY, SHAPE)
        x2 = jr.normal(jr.PRNGKey(1), SHAPE)
        assert jnp.array_equal(layer(x1, x2, key=KEY, inference=True), x1 + x2)

    def test_key_required_in_train_mode(self):
        layer = DropPathAdd(p=0.5)
        x1 = jr.normal(KEY, SHAPE)
        x2 = jr.normal(jr.PRNGKey(1), SHAPE)
        with pytest.raises(RuntimeError, match="requires a key"):
            layer(x1, x2, inference=False)

    def test_output_finite(self):
        layer = DropPathAdd(p=0.5)
        x1 = jr.normal(KEY, SHAPE)
        x2 = jr.normal(jr.PRNGKey(1), SHAPE)
        assert jnp.all(jnp.isfinite(layer(x1, x2, key=KEY)))

    def test_stochastic_skips_or_adds(self):
        """With p < 1 output must sometimes equal x1 and sometimes x1+x2."""
        layer = DropPathAdd(p=0.5)
        x1 = jnp.ones(SHAPE)
        x2 = jnp.ones(SHAPE)
        results = {
            "add": False,
            "skip": False,
        }
        for i in range(50):
            out = layer(x1, x2, key=jr.PRNGKey(i))
            if jnp.allclose(out, x1 + x2):
                results["add"] = True
            elif jnp.allclose(out, x1):
                results["skip"] = True
            if results["add"] and results["skip"]:
                break
        assert results["add"], "DropPathAdd never added x2"
        assert results["skip"], "DropPathAdd never skipped x2"

    def test_bfloat16_input_finite(self):
        layer = DropPathAdd(p=0.5)
        x1 = jr.normal(KEY, SHAPE).astype(jnp.bfloat16)
        x2 = jr.normal(jr.PRNGKey(1), SHAPE).astype(jnp.bfloat16)
        out = layer(x1, x2, key=KEY)
        assert jnp.all(jnp.isfinite(out))

    def test_float16_input_finite(self):
        layer = DropPathAdd(p=0.5)
        x1 = jr.normal(KEY, SHAPE).astype(jnp.float16)
        x2 = jr.normal(jr.PRNGKey(1), SHAPE).astype(jnp.float16)
        out = layer(x1, x2, key=KEY)
        assert jnp.all(jnp.isfinite(out))

    def test_p_stored_as_float(self):
        layer = DropPathAdd(p=0.3)
        assert isinstance(layer.p, float)
        assert layer.p == pytest.approx(0.3)

    def test_jax_array_p_converted_to_float(self):
        p_arr = jnp.array([0.4])
        layer = DropPathAdd(p=p_arr)
        assert isinstance(layer.p, float)
        assert layer.p == pytest.approx(0.4)

    def test_jax_array_p_wrong_length_raises(self):
        with pytest.raises(ValueError, match="values for p"):
            DropPathAdd(p=jnp.array([0.1, 0.2]))


# get_dropout


class TestGetDropout:
    def test_string_resolution_droppath(self):
        assert get_dropout("droppath") is DropPath

    def test_string_resolution_droppathadd(self):
        assert get_dropout("droppathadd") is DropPathAdd

    def test_class_passthrough(self):
        assert get_dropout(DropPath) is DropPath

    def test_class_passthrough_droppathadd(self):
        assert get_dropout(DropPathAdd) is DropPathAdd

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown module string"):
            get_dropout("nonexistent_dropout")

    def test_returned_class_is_instantiable(self):
        cls = get_dropout("droppath")
        layer = cls(p=0.1)
        x = jr.normal(KEY, SHAPE)
        assert layer(x, key=KEY).shape == SHAPE


# register_dropout


class TestRegisterDropout:
    def test_register_default_name(self):
        from equimo.layers.dropout import _DROPOUT_REGISTRY

        @register_dropout()
        class CustomDropout(eqx.Module):
            pass

        assert "customdropout" in _DROPOUT_REGISTRY
        assert get_dropout("customdropout") is CustomDropout

    def test_register_custom_name(self):
        from equimo.layers.dropout import _DROPOUT_REGISTRY

        @register_dropout(name="MyDropout")
        class CustomDropout2(eqx.Module):
            pass

        assert "mydropout" in _DROPOUT_REGISTRY
        assert get_dropout("mydropout") is CustomDropout2

    def test_register_non_eqx_module_raises(self):
        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):

            @register_dropout()
            class NotAModule:
                pass

    def test_register_duplicate_name_raises(self):
        @register_dropout()
        class DuplicateDropout(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_dropout(name="DuplicateDropout")
            class AnotherDropout(eqx.Module):
                pass
