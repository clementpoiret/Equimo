"""Tests for equimo.layers.norm."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.norm import (
    DyT,
    LayerNorm2d,
    LayerScale,
    RMSNorm2d,
    RMSNormGated,
    get_norm,
    register_norm,
)

# Shared fixtures

KEY = jr.PRNGKey(0)
DIM = 64
C, H, W = 32, 8, 8


# RMSNormGated


class TestRMSNormGated:
    def test_output_shape(self):
        layer = RMSNormGated(DIM)
        x = jr.normal(KEY, (DIM,))
        assert layer(x).shape == (DIM,)

    def test_output_finite(self):
        layer = RMSNormGated(DIM)
        x = jr.normal(KEY, (DIM,))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_gating_none_vs_ones_are_equal(self):
        """No gate and a gate of ones must produce the same output."""
        layer = RMSNormGated(DIM)
        x = jr.normal(KEY, (DIM,))
        out_no_gate = layer(x, z=None)
        out_ones = layer(x, z=jnp.ones(DIM))
        assert jnp.allclose(out_no_gate, out_ones, atol=1e-6)

    def test_gating_changes_output(self):
        """A non-trivial gate must change the output."""
        layer = RMSNormGated(DIM)
        x = jr.normal(KEY, (DIM,))
        z = jr.uniform(KEY, (DIM,)) + 0.1  # strictly positive, not ones
        out_no_gate = layer(x)
        out_gated = layer(x, z=z)
        assert not jnp.allclose(out_no_gate, out_gated)

    def test_zero_gate_gives_zero_output(self):
        """Gate of zeros collapses the input to zero, so output must be zero."""
        layer = RMSNormGated(DIM)
        x = jr.normal(KEY, (DIM,))
        out = layer(x, z=jnp.zeros(DIM))
        assert jnp.allclose(out, jnp.zeros(DIM), atol=1e-6)

    def test_rms_normalization_property(self):
        """With w=ones, mean(output²) ≈ 1 (unit RMS)."""
        layer = RMSNormGated(DIM)
        x = jr.normal(KEY, (DIM,))
        y = layer(x)
        assert jnp.isclose(jnp.mean(y**2), 1.0, atol=1e-4)

    def test_custom_eps(self):
        """Custom eps is stored and used; output must still be finite."""
        layer = RMSNormGated(DIM, eps=1e-3)
        assert layer.eps == 1e-3
        x = jr.normal(KEY, (DIM,))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_dtype_preserved_bfloat16(self):
        layer = RMSNormGated(DIM)
        x = jr.normal(KEY, (DIM,)).astype(jnp.bfloat16)
        out = layer(x)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_float16(self):
        layer = RMSNormGated(DIM)
        x = jr.normal(KEY, (DIM,)).astype(jnp.float16)
        out = layer(x)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))


# LayerScale


class TestLayerScale:
    def test_output_shape_1d(self):
        layer = LayerScale(DIM, axis=0)
        x = jr.normal(KEY, (DIM,))
        assert layer(x).shape == (DIM,)

    def test_output_shape_3d(self):
        layer = LayerScale(C, axis=0)
        x = jr.normal(KEY, (C, H, W))
        assert layer(x).shape == (C, H, W)

    def test_output_finite(self):
        layer = LayerScale(DIM)
        x = jr.normal(KEY, (DIM,))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_init_values_scale(self):
        """With init_values=v, output ≈ v * x."""
        init = 1e-2
        layer = LayerScale(DIM, init_values=init, axis=0)
        x = jr.normal(KEY, (DIM,))
        assert jnp.allclose(layer(x), init * x, atol=1e-6)

    def test_axis_last(self):
        """axis=-1 (last dim) must scale along the last dimension."""
        layer = LayerScale(DIM, init_values=2.0, axis=-1)
        x = jr.normal(KEY, (4, DIM))
        out = layer(x)
        assert out.shape == (4, DIM)
        assert jnp.allclose(out, 2.0 * x, atol=1e-6)

    def test_channel_mismatch_raises(self):
        layer = LayerScale(DIM, axis=0)
        x = jr.normal(KEY, (DIM + 1,))
        with pytest.raises(ValueError, match="Channel mismatch"):
            layer(x)

    def test_dtype_preserved_bfloat16(self):
        layer = LayerScale(DIM)
        x = jr.normal(KEY, (DIM,)).astype(jnp.bfloat16)
        out = layer(x)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_float16(self):
        layer = LayerScale(DIM)
        x = jr.normal(KEY, (DIM,)).astype(jnp.float16)
        out = layer(x)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))

    def test_custom_gamma_dtype(self):
        """gamma stored in bfloat16 must still produce a valid output."""
        layer = LayerScale(DIM, dtype=jnp.bfloat16)
        assert layer.gamma.dtype == jnp.bfloat16
        x = jr.normal(KEY, (DIM,))
        assert jnp.all(jnp.isfinite(layer(x)))


# DyT


class TestDyT:
    def test_output_shape(self):
        layer = DyT(DIM)
        x = jr.normal(KEY, (DIM,))
        assert layer(x).shape == (DIM,)

    def test_output_finite(self):
        layer = DyT(DIM)
        x = jr.normal(KEY, (DIM,))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_alpha_zero_gives_bias(self):
        """alpha=0 → tanh(0)=0 → output = 0 * weight + bias = bias."""
        layer = DyT(DIM, alpha_init_value=0.0)
        x = jr.normal(KEY, (DIM,))
        out = layer(x)
        # bias is initialized to zeros, so output must be zeros
        assert jnp.allclose(out, jnp.zeros(DIM), atol=1e-6)

    def test_alpha_init_value_stored(self):
        layer = DyT(DIM, alpha_init_value=0.3)
        assert jnp.allclose(layer.alpha, jnp.full((DIM,), 0.3))

    def test_weight_bias_initial_values(self):
        """weight starts at ones, bias at zeros."""
        layer = DyT(DIM)
        assert jnp.allclose(layer.weight, jnp.ones(DIM))
        assert jnp.allclose(layer.bias, jnp.zeros(DIM))

    def test_output_is_affine_transform_of_tanh(self):
        """output == tanh(alpha * x) * weight + bias by construction."""
        layer = DyT(DIM)
        x = jr.normal(KEY, (DIM,))
        expected = jnp.tanh(layer.alpha * x) * layer.weight + layer.bias
        assert jnp.allclose(layer(x), expected, atol=1e-6)

    def test_dtype_preserved_bfloat16(self):
        layer = DyT(DIM)
        x = jr.normal(KEY, (DIM,)).astype(jnp.bfloat16)
        out = layer(x)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_float16(self):
        layer = DyT(DIM)
        x = jr.normal(KEY, (DIM,)).astype(jnp.float16)
        out = layer(x)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))

    def test_large_input_stays_finite(self):
        """tanh saturates, so extreme inputs must not produce NaN/Inf."""
        layer = DyT(DIM)
        x = jnp.full((DIM,), 1e4)
        assert jnp.all(jnp.isfinite(layer(x)))


# RMSNorm2d


class TestRMSNorm2d:
    def test_output_shape(self):
        layer = RMSNorm2d(C)
        x = jr.normal(KEY, (C, H, W))
        assert layer(x).shape == (C, H, W)

    def test_output_finite(self):
        layer = RMSNorm2d(C)
        x = jr.normal(KEY, (C, H, W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_weight_shape_affine(self):
        layer = RMSNorm2d(C, affine=True)
        assert layer.weight is not None
        assert layer.weight.shape == (C,)

    def test_no_weight_when_no_affine(self):
        layer = RMSNorm2d(C, affine=False)
        assert layer.weight is None

    def test_no_affine_output_finite(self):
        layer = RMSNorm2d(C, affine=False)
        x = jr.normal(KEY, (C, H, W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_rms_normalization_property(self):
        """Without affine, mean of squared output over channels ≈ 1 at each (h, w)."""
        layer = RMSNorm2d(C, affine=False)
        x = jr.normal(KEY, (C, H, W))
        y = layer(x)
        rms_per_location = jnp.mean(y**2, axis=0)  # (H, W)
        assert jnp.allclose(rms_per_location, jnp.ones((H, W)), atol=1e-5)

    def test_custom_eps(self):
        layer = RMSNorm2d(C, eps=1e-3)
        assert layer.eps == 1e-3
        x = jr.normal(KEY, (C, H, W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_dtype_preserved_bfloat16(self):
        layer = RMSNorm2d(C)
        x = jr.normal(KEY, (C, H, W)).astype(jnp.bfloat16)
        out = layer(x)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_float16(self):
        layer = RMSNorm2d(C)
        x = jr.normal(KEY, (C, H, W)).astype(jnp.float16)
        out = layer(x)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))


# LayerNorm2d


class TestLayerNorm2d:
    def test_output_shape(self):
        layer = LayerNorm2d(C)
        x = jr.normal(KEY, (C, H, W))
        assert layer(x).shape == (C, H, W)

    def test_output_finite(self):
        layer = LayerNorm2d(C)
        x = jr.normal(KEY, (C, H, W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_weight_bias_shape_affine(self):
        layer = LayerNorm2d(C, affine=True)
        assert layer.weight is not None and layer.weight.shape == (C,)
        assert layer.bias is not None and layer.bias.shape == (C,)

    def test_no_weight_bias_when_no_affine(self):
        layer = LayerNorm2d(C, affine=False)
        assert layer.weight is None
        assert layer.bias is None

    def test_no_affine_output_finite(self):
        layer = LayerNorm2d(C, affine=False)
        x = jr.normal(KEY, (C, H, W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_mean_zero_property(self):
        """Without affine, mean over channels must be ≈ 0 at each spatial location."""
        layer = LayerNorm2d(C, affine=False)
        x = jr.normal(KEY, (C, H, W))
        y = layer(x)
        mean_per_location = jnp.mean(y, axis=0)  # (H, W)
        assert jnp.allclose(mean_per_location, jnp.zeros((H, W)), atol=1e-5)

    def test_unit_variance_property(self):
        """Without affine, variance over channels must be ≈ 1 at each spatial location."""
        layer = LayerNorm2d(C, affine=False)
        x = jr.normal(KEY, (C, H, W))
        y = layer(x)
        var_per_location = jnp.mean(y**2, axis=0)  # (H, W); mean already ≈ 0
        assert jnp.allclose(var_per_location, jnp.ones((H, W)), atol=1e-4)

    def test_custom_eps(self):
        layer = LayerNorm2d(C, eps=1e-3)
        assert layer.eps == 1e-3
        x = jr.normal(KEY, (C, H, W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_dtype_preserved_bfloat16(self):
        layer = LayerNorm2d(C)
        x = jr.normal(KEY, (C, H, W)).astype(jnp.bfloat16)
        out = layer(x)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_float16(self):
        layer = LayerNorm2d(C)
        x = jr.normal(KEY, (C, H, W)).astype(jnp.float16)
        out = layer(x)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))


# get_norm


class TestGetNorm:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("rmsnormgated", RMSNormGated),
            ("layerscale", LayerScale),
            ("dynamictanh", DyT),
            ("rmsnorm2d", RMSNorm2d),
            ("layernorm2d", LayerNorm2d),
            ("layernorm", eqx.nn.LayerNorm),
            ("rmsnorm", eqx.nn.RMSNorm),
            ("groupnorm", eqx.nn.GroupNorm),
        ],
    )
    def test_string_resolution(self, name, expected):
        assert get_norm(name) is expected

    def test_class_passthrough(self):
        assert get_norm(RMSNormGated) is RMSNormGated

    def test_class_passthrough_builtin(self):
        assert get_norm(eqx.nn.LayerNorm) is eqx.nn.LayerNorm

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown module string"):
            get_norm("nonexistent_norm")

    def test_returned_class_is_instantiable_rmsnormgated(self):
        cls = get_norm("rmsnormgated")
        layer = cls(DIM)
        x = jr.normal(KEY, (DIM,))
        assert layer(x).shape == (DIM,)

    def test_returned_class_is_instantiable_layernorm2d(self):
        cls = get_norm("layernorm2d")
        layer = cls(C)
        x = jr.normal(KEY, (C, H, W))
        assert layer(x).shape == (C, H, W)


# register_norm


class TestRegisterNorm:
    def test_register_default_name(self):
        from equimo.layers.norm import _NORM_REGISTRY

        @register_norm()
        class MyCustomNorm(eqx.Module):
            pass

        assert "mycustomnorm" in _NORM_REGISTRY
        assert get_norm("mycustomnorm") is MyCustomNorm

    def test_register_custom_name(self):
        from equimo.layers.norm import _NORM_REGISTRY

        @register_norm(name="SuperNorm42")
        class AnotherNorm(eqx.Module):
            pass

        assert "supernorm42" in _NORM_REGISTRY
        assert get_norm("supernorm42") is AnotherNorm

    def test_register_non_eqx_module_raises(self):
        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):

            @register_norm()
            class NotAModule:
                pass

    def test_register_duplicate_name_raises(self):
        @register_norm()
        class UniqueName(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_norm(name="UniqueName")
            class AnotherOne(eqx.Module):
                pass
