"""Tests for equimo.layers.downsample."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.downsample import (
    ConvNormDownsampler,
    PWSEDownsampler,
    _DOWNSAMPLER_REGISTRY,
    get_downsampler,
    register_downsampler,
)

KEY = jr.PRNGKey(0)
IN_CHANNELS = 32
OUT_CHANNELS = 64
H, W = 16, 16


# ---------------------------------------------------------------------------
# ConvNormDownsampler
# ---------------------------------------------------------------------------


class TestConvNormDownsampler:
    def test_simple_output_shape_default_out_channels(self):
        ds = ConvNormDownsampler(IN_CHANNELS, key=KEY)
        x = jnp.ones((IN_CHANNELS, H, W))
        out = ds(x)
        assert out.shape == (IN_CHANNELS * 2, H // 2, W // 2)

    def test_simple_output_shape_custom_out_channels(self):
        ds = ConvNormDownsampler(IN_CHANNELS, out_channels=OUT_CHANNELS, key=KEY)
        x = jnp.ones((IN_CHANNELS, H, W))
        out = ds(x)
        assert out.shape == (OUT_CHANNELS, H // 2, W // 2)

    def test_double_output_shape(self):
        """Double mode applies stride-2 twice → spatial dims quartered."""
        ds = ConvNormDownsampler(IN_CHANNELS, out_channels=OUT_CHANNELS, mode="double", key=KEY)
        x = jnp.ones((IN_CHANNELS, H, W))
        out = ds(x)
        assert out.shape == (OUT_CHANNELS, H // 4, W // 4)

    def test_simple_output_finite(self):
        ds = ConvNormDownsampler(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(ds(x)))

    def test_double_output_finite(self):
        ds = ConvNormDownsampler(IN_CHANNELS, out_channels=OUT_CHANNELS, mode="double", key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(ds(x)))

    def test_no_norm(self):
        ds = ConvNormDownsampler(IN_CHANNELS, use_norm=False, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(ds(x)))

    def test_use_bias(self):
        ds = ConvNormDownsampler(IN_CHANNELS, use_bias=True, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(ds(x)))

    def test_simple_with_act_layer(self):
        import jax

        ds = ConvNormDownsampler(IN_CHANNELS, act_layer=jax.nn.gelu, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        out = ds(x)
        assert out.shape == (IN_CHANNELS * 2, H // 2, W // 2)
        assert jnp.all(jnp.isfinite(out))

    def test_bfloat16_input(self):
        ds = ConvNormDownsampler(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(jnp.bfloat16)
        out = ds(x)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# PWSEDownsampler
# ---------------------------------------------------------------------------


class TestPWSEDownsampler:
    def test_output_shape(self):
        ds = PWSEDownsampler(IN_CHANNELS, OUT_CHANNELS, key=KEY)
        x = jnp.ones((IN_CHANNELS, H, W))
        out = ds(x, KEY)
        assert out.shape == (OUT_CHANNELS, H // 2, W // 2)

    def test_output_finite(self):
        ds = PWSEDownsampler(IN_CHANNELS, OUT_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(ds(x, KEY)))

    def test_deterministic_in_inference_mode(self):
        ds = PWSEDownsampler(IN_CHANNELS, OUT_CHANNELS, drop_path=0.5, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        out1 = ds(x, jr.PRNGKey(1), inference=True)
        out2 = ds(x, jr.PRNGKey(2), inference=True)
        assert jnp.allclose(out1, out2)

    def test_stochastic_in_train_mode(self):
        ds = PWSEDownsampler(IN_CHANNELS, OUT_CHANNELS, drop_path=0.5, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        out1 = ds(x, jr.PRNGKey(1), inference=False)
        out2 = ds(x, jr.PRNGKey(2), inference=False)
        assert not jnp.allclose(out1, out2)

    def test_kwargs_ignored(self):
        ds = PWSEDownsampler(IN_CHANNELS, OUT_CHANNELS, key=KEY, unknown_kwarg=True)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert ds(x, KEY).shape == (OUT_CHANNELS, H // 2, W // 2)

    def test_bfloat16_input(self):
        ds = PWSEDownsampler(IN_CHANNELS, OUT_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(jnp.bfloat16)
        out = ds(x, KEY, inference=True)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# get_downsampler
# ---------------------------------------------------------------------------


class TestGetDownsampler:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("convnormdownsampler", ConvNormDownsampler),
            ("pwsedownsampler", PWSEDownsampler),
        ],
    )
    def test_string_resolution(self, name, expected):
        assert get_downsampler(name) is expected

    def test_class_passthrough(self):
        assert get_downsampler(ConvNormDownsampler) is ConvNormDownsampler

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown module string"):
            get_downsampler("nonexistent_ds")

    def test_returned_class_is_instantiable(self):
        cls = get_downsampler("convnormdownsampler")
        model = cls(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        out = model(x)
        assert out.shape == (IN_CHANNELS * 2, H // 2, W // 2)


# ---------------------------------------------------------------------------
# register_downsampler
# ---------------------------------------------------------------------------


class TestRegisterDownsampler:
    def test_register_default_name(self):
        import equinox as eqx

        @register_downsampler()
        class CustomDS(eqx.Module):
            pass

        assert "customds" in _DOWNSAMPLER_REGISTRY
        assert get_downsampler("customds") is CustomDS

    def test_register_custom_name(self):
        import equinox as eqx

        @register_downsampler(name="MySuperDS")
        class CustomDS2(eqx.Module):
            pass

        assert "mysuperds" in _DOWNSAMPLER_REGISTRY
        assert get_downsampler("mysuperds") is CustomDS2

    def test_register_non_eqx_module(self):
        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):

            @register_downsampler()
            class NotAModule:
                pass

    def test_register_duplicate_name(self):
        import equinox as eqx

        @register_downsampler()
        class DuplicateDS(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_downsampler(name="DuplicateDS")
            class AnotherDS(eqx.Module):
                pass
