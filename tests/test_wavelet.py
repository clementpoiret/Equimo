"""Tests for equimo.layers.wavelet."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.wavelet import (
    HWDConv,
    _depthwise_conv2d_stride2,
    _haar_1d,
    _haar_2d_kernels,
    _WAVELET_REGISTRY,
    get_wavelet,
    haar_dwt_split,
    register_wavelet,
)

KEY = jr.PRNGKey(0)


# ---------------------------------------------------------------------------
# Haar wavelet helpers
# ---------------------------------------------------------------------------


class TestHaarHelpers:
    def test_haar_1d_shapes(self):
        h0, h1 = _haar_1d()
        assert h0.shape == (2,)
        assert h1.shape == (2,)

    def test_haar_1d_orthonormal(self):
        """h0 and h1 must be unit norm and orthogonal to each other."""
        h0, h1 = _haar_1d()
        assert jnp.allclose(jnp.dot(h0, h0), jnp.array(1.0), atol=1e-6)
        assert jnp.allclose(jnp.dot(h1, h1), jnp.array(1.0), atol=1e-6)
        assert jnp.allclose(jnp.dot(h0, h1), jnp.array(0.0), atol=1e-6)

    def test_haar_2d_kernels_shapes(self):
        kLL, kHL, kLH, kHH = _haar_2d_kernels()
        for k in (kLL, kHL, kLH, kHH):
            assert k.shape == (2, 2)

    def test_depthwise_conv2d_shape(self):
        x = jnp.ones((8, 16, 16))
        kLL, _, _, _ = _haar_2d_kernels()
        out = _depthwise_conv2d_stride2(x, kLL)
        assert out.shape == (8, 8, 8)

    def test_haar_dwt_split_shapes(self):
        x = jnp.ones((4, 32, 32))
        LL, HL, LH, HH = haar_dwt_split(x)
        for band in (LL, HL, LH, HH):
            assert band.shape == (4, 16, 16)

    def test_haar_dwt_split_energy_conservation(self):
        """Reconstruction from LL + detail bands must approximately preserve energy."""
        x = jr.normal(KEY, (4, 8, 8))
        LL, HL, LH, HH = haar_dwt_split(x)
        # Each band has 1/4 the spatial area; 4 bands restore original energy
        total_energy = sum(jnp.sum(b**2) for b in (LL, HL, LH, HH))
        assert jnp.allclose(total_energy, jnp.sum(x**2), rtol=1e-4)

    def test_haar_dwt_split_odd_size_no_crash(self):
        """haar_dwt_split should not crash on odd spatial dimensions (SAME padding)."""
        x = jr.normal(KEY, (4, 17, 15))
        LL, HL, LH, HH = haar_dwt_split(x)
        assert LL.shape[0] == 4


# ---------------------------------------------------------------------------
# HWDConv
# ---------------------------------------------------------------------------


class TestHWDConv:
    @pytest.mark.parametrize("mode", ["h_discard", "band_grouped", "accurate"])
    def test_output_shape(self, mode):
        in_c, out_c = 8, 16 if mode == "band_grouped" else 12
        layer = HWDConv(in_c, out_c, mode=mode, key=KEY)
        x = jr.normal(KEY, (in_c, 32, 32))
        out = layer(x, key=KEY)
        assert out.shape == (out_c, 16, 16)

    def test_halves_spatial_dims(self):
        layer = HWDConv(4, 8, key=KEY)
        x = jr.normal(KEY, (4, 64, 64))
        out = layer(x, key=KEY)
        assert out.shape == (8, 32, 32)

    def test_output_finite_float32(self):
        layer = HWDConv(4, 8, key=KEY)
        x = jr.normal(KEY, (4, 16, 16))
        assert jnp.all(jnp.isfinite(layer(x, key=KEY)))

    def test_bfloat16_input_finite(self):
        layer = HWDConv(4, 8, key=KEY)
        x = jr.normal(KEY, (4, 16, 16)).astype(jnp.bfloat16)
        out = layer(x, key=KEY)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_float16_input_finite(self):
        layer = HWDConv(4, 8, key=KEY)
        x = jr.normal(KEY, (4, 16, 16)).astype(jnp.float16)
        out = layer(x, key=KEY)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))

    def test_output_dtype_preserved(self):
        layer = HWDConv(4, 8, key=KEY)
        for dtype in (jnp.float32, jnp.bfloat16):
            x = jr.normal(KEY, (4, 16, 16)).astype(dtype)
            assert layer(x, key=KEY).dtype == dtype

    def test_inference_mode_deterministic(self):
        layer = HWDConv(4, 8, dropout=0.5, key=KEY)
        x = jr.normal(KEY, (4, 16, 16))
        out1 = layer(x, key=jr.PRNGKey(1), inference=True)
        out2 = layer(x, key=jr.PRNGKey(2), inference=True)
        assert jnp.array_equal(out1, out2)

    def test_band_grouped_out_not_divisible_by_4_raises(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            HWDConv(4, 6, mode="band_grouped", key=KEY)

    def test_h_discard_mode_shape(self):
        """h_discard keeps only LL subband — input channels unchanged, spatial halved."""
        layer = HWDConv(8, 16, mode="h_discard", key=KEY)
        x = jr.normal(KEY, (8, 32, 32))
        assert layer(x, key=KEY).shape == (16, 16, 16)

    def test_jit_compatible(self):
        layer = HWDConv(4, 8, key=KEY)
        x = jr.normal(KEY, (4, 16, 16))
        fn = eqx.filter_jit(lambda m, x: m(x, key=KEY))
        out = fn(layer, x)
        assert out.shape == (8, 8, 8)


# ---------------------------------------------------------------------------
# get_wavelet / register_wavelet
# ---------------------------------------------------------------------------


class TestGetWavelet:
    def test_string_hwdconv(self):
        assert get_wavelet("hwdconv") is HWDConv

    def test_class_passthrough(self):
        assert get_wavelet(HWDConv) is HWDConv

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown module string"):
            get_wavelet("nonexistent_wavelet")

    def test_returned_class_instantiable(self):
        cls = get_wavelet("hwdconv")
        layer = cls(4, 8, key=KEY)
        x = jr.normal(KEY, (4, 16, 16))
        assert layer(x, key=KEY).shape == (8, 8, 8)


class TestRegisterWavelet:
    def test_register_default_name(self):
        @register_wavelet()
        class MyWavelet(eqx.Module):
            pass

        assert "mywavelet" in _WAVELET_REGISTRY
        assert get_wavelet("mywavelet") is MyWavelet

    def test_register_custom_name(self):
        @register_wavelet(name="CustomWave")
        class AnotherWavelet(eqx.Module):
            pass

        assert "customwave" in _WAVELET_REGISTRY
        assert get_wavelet("customwave") is AnotherWavelet

    def test_register_non_eqx_module_raises(self):
        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):
            @register_wavelet()
            class NotAModule:
                pass

    def test_register_duplicate_raises(self):
        @register_wavelet(name="dup_wavelet")
        class W1(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):
            @register_wavelet(name="dup_wavelet")
            class W2(eqx.Module):
                pass
