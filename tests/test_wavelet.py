"""Tests for equimo.layers.wavelet."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.wavelet import (
    _WAVELET_REGISTRY,
    HWDConv,
    _depthwise_conv2d_stride2,
    _haar_1d,
    _haar_2d_kernels,
    get_wavelet,
    haar_dwt_split,
    haar_dwt_split_conv,
    inverse_haar_dwt_split,
    register_wavelet,
)

KEY = jr.PRNGKey(0)


# Haar wavelet helpers


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
        LL, HL, LH, HH = haar_dwt_split_conv(x)
        for band in (LL, HL, LH, HH):
            assert band.shape == (4, 16, 16)

    def test_haar_dwt_split_energy_conservation(self):
        """Reconstruction from LL + detail bands must approximately preserve energy."""
        x = jr.normal(KEY, (4, 8, 8))
        LL, HL, LH, HH = haar_dwt_split_conv(x)
        # Each band has 1/4 the spatial area; 4 bands restore original energy
        total_energy = sum(jnp.sum(b**2) for b in (LL, HL, LH, HH))
        assert jnp.allclose(total_energy, jnp.sum(x**2), rtol=1e-4)

    def test_haar_dwt_split_odd_size_no_crash(self):
        """haar_dwt_split should not crash on odd spatial dimensions (SAME padding)."""
        x = jr.normal(KEY, (4, 17, 15))
        LL, HL, LH, HH = haar_dwt_split_conv(x)
        assert LL.shape[0] == 4


class TestHaarLinear:
    """Tests for the polyphase (slicing) Haar DWT implementation."""

    def test_shapes_even(self):
        x = jnp.ones((8, 16, 16))
        LL, HL, LH, HH = haar_dwt_split(x)
        for band in (LL, HL, LH, HH):
            assert band.shape == (8, 8, 8)

    def test_shapes_odd(self):
        x = jr.normal(KEY, (4, 17, 15))
        LL, HL, LH, HH = haar_dwt_split(x)
        # ceil(17/2)=9, ceil(15/2)=8
        for band in (LL, HL, LH, HH):
            assert band.shape == (4, 9, 8)

    def test_energy_conservation(self):
        """Parseval's theorem: sum of squared sub-band coefficients == input energy."""
        x = jr.normal(KEY, (4, 8, 8))
        LL, HL, LH, HH = haar_dwt_split(x)
        total_energy = sum(jnp.sum(b**2) for b in (LL, HL, LH, HH))
        assert jnp.allclose(total_energy, jnp.sum(x**2), rtol=1e-4)

    def test_dtype_preserved(self):
        for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
            x = jr.normal(KEY, (2, 8, 8)).astype(dtype)
            LL, _, _, _ = haar_dwt_split(x, dtype=dtype)
            assert LL.dtype == dtype

    def test_constant_input_ll_only(self):
        """A spatially constant signal has zero energy in all detail bands."""
        x = jnp.ones((2, 8, 8)) * 3.0
        LL, HL, LH, HH = haar_dwt_split(x)
        assert jnp.allclose(HL, 0.0, atol=1e-7)
        assert jnp.allclose(LH, 0.0, atol=1e-7)
        assert jnp.allclose(HH, 0.0, atol=1e-7)
        assert jnp.allclose(LL, 6.0, atol=1e-6)

    def test_jit_compatible(self):
        x = jr.normal(KEY, (4, 16, 16))
        fn = jax.jit(lambda x: haar_dwt_split(x))
        LL, HL, LH, HH = fn(x)
        assert LL.shape == (4, 8, 8)

    def test_minimal_input(self):
        """1×1 spatial input (pad to 2×2)."""
        x = jnp.ones((1, 1, 1)) * 2.0
        LL, HL, LH, HH = haar_dwt_split(x)
        assert LL.shape == (1, 1, 1)

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 2, 2),
            (4, 8, 8),
            (8, 16, 16),
            (4, 32, 32),
            (4, 17, 15),
            (2, 1, 1),
            (3, 7, 11),
        ],
    )
    def test_equivalence_conv_vs_linear(self, shape):
        """Linear and conv implementations must produce matching results."""
        x = jr.normal(KEY, shape)
        conv_bands = haar_dwt_split_conv(x)
        linear_bands = haar_dwt_split(x)
        for c, l in zip(conv_bands, linear_bands):
            assert c.shape == l.shape
            assert jnp.allclose(c, l, atol=1e-6, rtol=1e-5), (
                f"Max abs diff: {float(jnp.max(jnp.abs(c - l)))}"
            )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_equivalence_conv_vs_linear_dtypes(self, dtype):
        x = jr.normal(KEY, (4, 16, 16)).astype(dtype)
        conv_bands = haar_dwt_split_conv(x, dtype=dtype)
        linear_bands = haar_dwt_split(x, dtype=dtype)
        atol = 1e-6 if dtype == jnp.float32 else 8e-2
        for c, l in zip(conv_bands, linear_bands):
            assert jnp.allclose(c, l, atol=atol), (
                f"dtype={dtype}, max diff: {float(jnp.max(jnp.abs(c - l)))}"
            )

    def test_hwdconv_uses_linear_and_matches_conv(self):
        """End-to-end: HWDConv output with linear backend ≈ manual conv-based split."""
        in_c, out_c = 4, 8
        layer = HWDConv(in_c, out_c, mode="accurate", key=KEY)
        x = jr.normal(KEY, (in_c, 16, 16))

        # Current HWDConv uses haar_dwt_split_linear internally
        out_layer = layer(x, key=KEY)

        # Manually reproduce with conv-based split
        LL, HL, LH, HH = haar_dwt_split_conv(x, dtype=x.dtype)
        y_conv = jnp.concatenate([LL, HL, LH, HH], axis=0)
        y_conv = layer.pre_norm(y_conv)
        y_conv = layer.proj(y_conv)
        y_conv = layer.act(y_conv)
        # dropout is 0.0 by default, so no key needed for comparison

        assert jnp.allclose(out_layer, y_conv, atol=1e-5), (
            f"Max diff: {float(jnp.max(jnp.abs(out_layer - y_conv)))}"
        )


class TestInverseHaarLinear:
    """Tests for the inverse polyphase Haar DWT."""

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 2, 2),
            (4, 8, 8),
            (8, 16, 16),
            (4, 32, 32),
            (2, 64, 64),
        ],
    )
    def test_perfect_reconstruction_even(self, shape):
        """Forward → inverse must recover the original signal (even dims)."""
        x = jr.normal(KEY, shape)
        bands = haar_dwt_split(x)
        x_rec = inverse_haar_dwt_split(*bands, dtype=x.dtype)
        assert x_rec.shape == x.shape
        assert jnp.allclose(x_rec, x, atol=1e-6), (
            f"Max abs err: {float(jnp.max(jnp.abs(x_rec - x)))}"
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 17, 15),
            (3, 7, 11),
            (2, 1, 1),
            (1, 3, 5),
            (2, 9, 8),
            (2, 8, 9),
        ],
    )
    def test_perfect_reconstruction_odd(self, shape):
        """Forward → inverse with orig dims must recover the signal (odd dims)."""
        C, H, W = shape
        x = jr.normal(KEY, shape)
        bands = haar_dwt_split(x)
        x_rec = inverse_haar_dwt_split(*bands, orig_h=H, orig_w=W, dtype=x.dtype)
        assert x_rec.shape == x.shape
        assert jnp.allclose(x_rec, x, atol=1e-6), (
            f"Max abs err: {float(jnp.max(jnp.abs(x_rec - x)))}"
        )

    def test_reconstruction_without_crop_oversized(self):
        """Without orig dims, inverse of an odd-dim input is zero-padded at boundary."""
        x = jr.normal(KEY, (2, 7, 5))
        bands = haar_dwt_split(x)
        x_rec = inverse_haar_dwt_split(*bands, dtype=x.dtype)
        assert x_rec.shape == (2, 8, 6)
        assert jnp.allclose(x_rec[:, :7, :5], x, atol=1e-6)

    def test_constant_signal_roundtrip(self):
        x = jnp.ones((2, 8, 8)) * 3.0
        bands = haar_dwt_split(x)
        x_rec = inverse_haar_dwt_split(*bands, dtype=x.dtype)
        assert jnp.allclose(x_rec, x, atol=1e-7)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_dtype_preserved(self, dtype):
        x = jr.normal(KEY, (4, 16, 16)).astype(dtype)
        bands = haar_dwt_split(x, dtype=dtype)
        x_rec = inverse_haar_dwt_split(*bands, dtype=dtype)
        assert x_rec.dtype == dtype

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_roundtrip_dtype_accuracy(self, dtype):
        """Roundtrip stays within dtype-appropriate tolerance."""
        x = jr.normal(KEY, (4, 16, 16)).astype(dtype)
        bands = haar_dwt_split(x, dtype=dtype)
        x_rec = inverse_haar_dwt_split(*bands, dtype=dtype)
        atol = 1e-6 if dtype == jnp.float32 else 5e-2
        assert jnp.allclose(x_rec, x, atol=atol), (
            f"dtype={dtype}, max diff: {float(jnp.max(jnp.abs(x_rec - x)))}"
        )

    def test_inverse_of_conv_forward(self):
        """Inverse of linear impl must also invert the conv-based forward."""
        x = jr.normal(KEY, (4, 8, 8))
        conv_bands = haar_dwt_split_conv(x)
        x_rec = inverse_haar_dwt_split(*conv_bands, dtype=x.dtype)
        assert jnp.allclose(x_rec, x, atol=1e-5), (
            f"Max abs err: {float(jnp.max(jnp.abs(x_rec - x)))}"
        )

    def test_jit_compatible(self):
        x = jr.normal(KEY, (4, 16, 16))
        bands = haar_dwt_split(x)

        @jax.jit
        def roundtrip(LL, HL, LH, HH):
            return inverse_haar_dwt_split(LL, HL, LH, HH)

        x_rec = roundtrip(*bands)
        assert jnp.allclose(x_rec, x, atol=1e-6)

    def test_minimal_input(self):
        """1×1 spatial → pad to 2×2, sub-bands are 1×1, inverse recovers 1×1."""
        x = jnp.array([[[5.0]]])
        bands = haar_dwt_split(x)
        x_rec = inverse_haar_dwt_split(*bands, orig_h=1, orig_w=1, dtype=x.dtype)
        assert x_rec.shape == (1, 1, 1)
        assert jnp.allclose(x_rec, x, atol=1e-6)

    def test_energy_conservation_inverse(self):
        """Orthogonal transform: energy(sub-bands) == energy(reconstruction)."""
        x = jr.normal(KEY, (4, 8, 8))
        bands = haar_dwt_split(x)
        band_energy = sum(jnp.sum(b**2) for b in bands)
        x_rec = inverse_haar_dwt_split(*bands, dtype=x.dtype)
        rec_energy = jnp.sum(x_rec**2)
        assert jnp.allclose(band_energy, rec_energy, rtol=1e-5)


# HWDConv


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
        dtype = jnp.bfloat16
        layer = jax.tree_util.tree_map(
            lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
            layer,
        )
        x = jr.normal(KEY, (4, 16, 16)).astype(dtype)
        out = layer(x, key=KEY)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_float16_input_finite(self):
        layer = HWDConv(4, 8, key=KEY)
        dtype = jnp.float16
        layer = jax.tree_util.tree_map(
            lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
            layer,
        )
        x = jr.normal(KEY, (4, 16, 16)).astype(dtype)
        out = layer(x, key=KEY)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))

    def test_output_dtype_preserved(self):
        for dtype in (jnp.float32, jnp.bfloat16):
            layer = HWDConv(4, 8, key=KEY)
            layer = jax.tree_util.tree_map(
                lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
                layer,
            )
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


# get_wavelet / register_wavelet


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
