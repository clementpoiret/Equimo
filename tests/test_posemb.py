"""Tests for equimo.layers.posemb."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.posemb import (
    DinoRoPE,
    LearnedPosEmbed,
    PosCNN,
    PosCNN2D,
    PosEmbMLPSwinv1D,
    PosEmbMLPSwinv2D,
    RoPE,
    VisionRoPE,
    get_posemb,
    register_posemb,
)

# Shared fixtures

KEY = jr.PRNGKey(0)
DIM = 64
NUM_HEADS = 4
H, W = 8, 8
SEQLEN = H * W  # 64


# LearnedPosEmbed


class TestLearnedPosEmbed:
    def _make(self, num_patches=64, dim=DIM, embed_size=8):
        return LearnedPosEmbed(
            weight=jr.normal(KEY, (num_patches, dim)),
            dim=dim,
            embed_size=embed_size,
            num_prefix_tokens=0,
            num_embedded_prefix_tokens=0,
            no_embed_class=True,
            pos_embed_reg_tokens=False,
        )

    def test_no_embed_class_shape(self):
        pe = self._make()
        x = jr.normal(KEY, (SEQLEN, DIM))
        out = pe(x, cls_token=None, reg_tokens=None, dynamic_img_size=False)
        assert out.shape == (SEQLEN, DIM)

    def test_no_embed_class_with_cls(self):
        pe = LearnedPosEmbed(
            weight=jr.normal(KEY, (SEQLEN, DIM)),
            dim=DIM,
            embed_size=H,
            num_prefix_tokens=1,
            num_embedded_prefix_tokens=0,
            no_embed_class=True,
            pos_embed_reg_tokens=False,
        )
        x = jr.normal(KEY, (SEQLEN, DIM))
        cls = jr.normal(KEY, (1, DIM))
        out = pe(x, cls_token=cls, reg_tokens=None, dynamic_img_size=False)
        assert out.shape == (SEQLEN + 1, DIM)

    def test_dynamic_img_size_same_size(self):
        pe = self._make()
        x = jr.normal(KEY, (DIM, H, W))
        out = pe(x, cls_token=None, reg_tokens=None, dynamic_img_size=True)
        assert out.shape == (SEQLEN, DIM)

    def test_dynamic_img_size_different_size(self):
        """Resample to a different spatial size."""
        pe = self._make(num_patches=64, embed_size=8)
        x = jr.normal(KEY, (DIM, 4, 4))
        out = pe(x, cls_token=None, reg_tokens=None, dynamic_img_size=True)
        assert out.shape == (16, DIM)

    def test_channel_mismatch_raises(self):
        pe = self._make()
        x = jr.normal(KEY, (DIM + 4, H, W))
        with pytest.raises(ValueError, match="Channel dim mismatch"):
            pe(x, cls_token=None, reg_tokens=None, dynamic_img_size=True)

    def test_reg_without_cls_raises(self):
        pe = LearnedPosEmbed(
            weight=jr.normal(KEY, (SEQLEN, DIM)),
            dim=DIM,
            embed_size=H,
            num_prefix_tokens=0,
            num_embedded_prefix_tokens=0,
            no_embed_class=False,
            pos_embed_reg_tokens=False,
        )
        x = jr.normal(KEY, (SEQLEN, DIM))
        reg = jr.normal(KEY, (2, DIM))
        with pytest.raises(ValueError, match="reg_tokens without cls_token"):
            pe(x, cls_token=None, reg_tokens=reg, dynamic_img_size=False)

    def test_output_finite(self):
        pe = self._make()
        x = jr.normal(KEY, (SEQLEN, DIM))
        out = pe(x, cls_token=None, reg_tokens=None, dynamic_img_size=False)
        assert jnp.all(jnp.isfinite(out))

    def test_bfloat16_input_finite(self):
        pe = self._make()
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(jnp.bfloat16)
        out = pe(x, cls_token=None, reg_tokens=None, dynamic_img_size=False)
        assert jnp.all(jnp.isfinite(out))

    def test_resample_returns_same_when_equal_size(self):
        pe = self._make(num_patches=64, embed_size=8)
        resampled = pe.resample(new_size=(8, 8), old_size=(8, 8))
        assert resampled.shape == (64, DIM)


# PosEmbMLPSwinv1D


class TestPosEmbMLPSwinv1D:
    def test_output_shape_rank1(self):
        layer = PosEmbMLPSwinv1D(DIM, rank=1, seq_len=SEQLEN, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert layer(x).shape == (SEQLEN, DIM)

    def test_output_shape_rank2(self):
        layer = PosEmbMLPSwinv1D(DIM, rank=2, seq_len=SEQLEN, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert layer(x).shape == (SEQLEN, DIM)

    def test_output_finite_rank1(self):
        layer = PosEmbMLPSwinv1D(DIM, rank=1, seq_len=SEQLEN, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_output_finite_rank2(self):
        layer = PosEmbMLPSwinv1D(DIM, rank=2, seq_len=SEQLEN, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_bfloat16_input_finite(self):
        """pos_emb is cast to input dtype before addition."""
        layer = PosEmbMLPSwinv1D(DIM, rank=1, seq_len=SEQLEN, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(jnp.bfloat16)
        out = layer(x)
        assert jnp.all(jnp.isfinite(out))

    def test_output_changes_input(self):
        """Adding pos encoding must change the input."""
        layer = PosEmbMLPSwinv1D(DIM, rank=1, seq_len=SEQLEN, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert not jnp.allclose(layer(x), x)

    def test_coords_table_shape_rank1(self):
        layer = PosEmbMLPSwinv1D(DIM, rank=1, seq_len=SEQLEN, key=KEY)
        assert layer.relative_coords_table.shape == (SEQLEN, 1)

    def test_coords_table_shape_rank2(self):
        layer = PosEmbMLPSwinv1D(DIM, rank=2, seq_len=SEQLEN, key=KEY)
        # meshgrid of (sqrt(SEQLEN), sqrt(SEQLEN)) stacked → (2, sqrt(SEQLEN), sqrt(SEQLEN))
        side = int(SEQLEN**0.5)
        assert layer.relative_coords_table.shape == (2, side, side)


# PosEmbMLPSwinv2D


class TestPosEmbMLPSwinv2D:
    def _make(self, window_size=(4, 4), pretrained_window_size=(4, 4)):
        return PosEmbMLPSwinv2D(
            window_size=window_size,
            pretrained_window_size=pretrained_window_size,
            num_heads=NUM_HEADS,
            seq_len=window_size[0] * window_size[1],
            key=KEY,
        )

    def test_output_shape(self):
        layer = self._make()
        ws = 4 * 4
        x = jr.normal(KEY, (NUM_HEADS, ws, ws))
        out = layer(x, local_window_size=ws)
        assert out.shape == (NUM_HEADS, ws, ws)

    def test_output_finite(self):
        layer = self._make()
        ws = 4 * 4
        x = jr.normal(KEY, (NUM_HEADS, ws, ws))
        assert jnp.all(jnp.isfinite(layer(x, local_window_size=ws)))

    def test_bfloat16_input_finite(self):
        """bias is cast to input dtype before addition."""
        layer = self._make()
        ws = 4 * 4
        x = jr.normal(KEY, (NUM_HEADS, ws, ws)).astype(jnp.bfloat16)
        out = layer(x, local_window_size=ws)
        assert jnp.all(jnp.isfinite(out))

    def test_pretrained_window_size_zero(self):
        """pretrained_window_size=(0,0) triggers fallback normalization."""
        layer = PosEmbMLPSwinv2D(
            window_size=(4, 4),
            pretrained_window_size=(0, 0),
            num_heads=NUM_HEADS,
            seq_len=16,
            key=KEY,
        )
        ws = 4 * 4
        x = jr.normal(KEY, (NUM_HEADS, ws, ws))
        assert jnp.all(jnp.isfinite(layer(x, local_window_size=ws)))

    def test_no_log_flag(self):
        layer = PosEmbMLPSwinv2D(
            window_size=(4, 4),
            pretrained_window_size=(4, 4),
            num_heads=NUM_HEADS,
            seq_len=16,
            key=KEY,
            no_log=True,
        )
        ws = 4 * 4
        x = jr.normal(KEY, (NUM_HEADS, ws, ws))
        assert jnp.all(jnp.isfinite(layer(x, local_window_size=ws)))

    def test_relative_position_index_shape(self):
        layer = self._make()
        ws = 4
        assert layer.relative_position_index.shape == (ws * ws, ws * ws)


# RoPE


class TestRoPE:
    def test_output_shape_1d(self):
        rope = RoPE(shape=(SEQLEN, DIM))
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert rope(x).shape == (SEQLEN, DIM)

    def test_output_shape_2d(self):
        rope = RoPE(shape=(H, W, DIM))
        x = jr.normal(KEY, (H, W, DIM))
        assert rope(x).shape == (H, W, DIM)

    def test_output_finite(self):
        rope = RoPE(shape=(SEQLEN, DIM))
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(rope(x)))

    def test_dtype_preserved_bfloat16(self):
        rope = RoPE(shape=(SEQLEN, DIM))
        dtype = jnp.bfloat16
        rope = jax.tree_util.tree_map(
            lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
            rope,
        )
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(dtype)
        out = rope(x)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_float16(self):
        rope = RoPE(shape=(SEQLEN, DIM))
        dtype = jnp.float16
        rope = jax.tree_util.tree_map(
            lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
            rope,
        )
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(dtype)
        out = rope(x)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))

    def test_rotations_shape(self):
        rope = RoPE(shape=(H, W, DIM))
        # angles: concat of len(channel_dims) tensors of shape (H, W, k_max) → (H, W, 2*k_max)
        # rotations: stack(re, im) → (H, W, 2*k_max, 2)
        k_max = DIM // (2 * 2)  # feature_dim // (2 * num_channel_dims)
        assert rope.rotations.shape == (H, W, 2 * k_max, 2)

    def test_output_changes_input(self):
        rope = RoPE(shape=(SEQLEN, DIM))
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert not jnp.allclose(rope(x), x)

    def test_feature_dim_not_divisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            # DIM=64 shape (4, 5) → k_max = 5 // (2*1) = 2, but 5 % 2 != 0
            RoPE(shape=(4, 5))


# DinoRoPE


class TestDinoRoPE:
    def _make(self, **kwargs):
        return DinoRoPE(DIM, num_heads=NUM_HEADS, **kwargs)

    def test_sincos_shape(self):
        rope = self._make()
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        d_head = DIM // NUM_HEADS
        assert sin.shape == (H * W, d_head)
        assert cos.shape == (H * W, d_head)

    def test_sincos_finite(self):
        rope = self._make()
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.all(jnp.isfinite(sin))
        assert jnp.all(jnp.isfinite(cos))

    def test_sincos_values_bounded(self):
        """sin/cos values must stay in [-1, 1]."""
        rope = self._make()
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.all(jnp.abs(sin) <= 1.0 + 1e-5)
        assert jnp.all(jnp.abs(cos) <= 1.0 + 1e-5)

    def test_deterministic_in_inference(self):
        """No augmentations in inference; output must be key-independent."""
        rope = self._make(shift_coords=0.1, jitter_coords=1.5, rescale_coords=1.5)
        sin1, cos1 = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(1), inference=True)
        sin2, cos2 = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(2), inference=True)
        assert jnp.allclose(sin1, sin2)
        assert jnp.allclose(cos1, cos2)

    def test_stochastic_in_training(self):
        """Training augmentations must produce different outputs for different keys."""
        rope = self._make(shift_coords=0.5)
        sin1, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(1), inference=False)
        sin2, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(2), inference=False)
        assert not jnp.allclose(sin1, sin2)

    def test_dim_not_divisible_raises(self):
        with pytest.raises(ValueError, match="4 \\* num_heads"):
            DinoRoPE(65, num_heads=4)

    def test_base_and_periods_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            DinoRoPE(
                DIM, num_heads=NUM_HEADS, base=100.0, min_period=1.0, max_period=10.0
            )

    def test_neither_base_nor_periods_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            DinoRoPE(DIM, num_heads=NUM_HEADS, base=None)

    def test_min_max_period_mode(self):
        rope = DinoRoPE(
            DIM, num_heads=NUM_HEADS, base=None, min_period=1.0, max_period=100.0
        )
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.all(jnp.isfinite(sin))

    @pytest.mark.parametrize("normalize_coords", ["min", "max", "separate"])
    def test_normalize_coords_modes(self, normalize_coords):
        rope = self._make(normalize_coords=normalize_coords)
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.all(jnp.isfinite(sin))

    def test_invalid_normalize_coords_raises(self):
        with pytest.raises(ValueError, match="Unknown normalize_coords"):
            DinoRoPE(DIM, num_heads=NUM_HEADS, normalize_coords="invalid")

    def test_rescale_augmentation(self):
        rope = self._make(rescale_coords=2.0)
        sin1, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(10), inference=False)
        sin2, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(20), inference=False)
        assert not jnp.allclose(sin1, sin2)

    def test_jitter_augmentation(self):
        rope = self._make(jitter_coords=2.0)
        sin1, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(10), inference=False)
        sin2, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(20), inference=False)
        assert not jnp.allclose(sin1, sin2)

    def test_periods_shape(self):
        rope = self._make()
        d_quarter = (DIM // NUM_HEADS) // 4
        assert rope.periods.shape == (d_quarter,)


# VisionRoPE


class TestVisionRoPE:
    """Tests for the unified VisionRoPE module."""

    # -- construction helpers ------------------------------------------------

    def _make_period(self, **kwargs):
        defaults = dict(strategy="period", dim=DIM, num_heads=NUM_HEADS)
        defaults.update(kwargs)
        return VisionRoPE(**defaults)

    def _make_mode(self, **kwargs):
        defaults = dict(strategy="mode", dim=DIM // 2)
        defaults.update(kwargs)
        return VisionRoPE(**defaults)

    # -- strategy="period": shapes and values --------------------------------

    def test_period_sincos_shape(self):
        rope = self._make_period()
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        d_head = DIM // NUM_HEADS
        assert sin.shape == (H * W, d_head)
        assert cos.shape == (H * W, d_head)

    def test_period_sincos_finite(self):
        rope = self._make_period()
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.all(jnp.isfinite(sin))
        assert jnp.all(jnp.isfinite(cos))

    def test_period_sincos_bounded(self):
        rope = self._make_period()
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.all(jnp.abs(sin) <= 1.0 + 1e-5)
        assert jnp.all(jnp.abs(cos) <= 1.0 + 1e-5)

    def test_period_call_shape(self):
        rope = self._make_period()
        d_head = DIM // NUM_HEADS
        x = jr.normal(KEY, (H * W, NUM_HEADS, d_head))
        out = rope(x, key=KEY, inference=True)
        assert out.shape == x.shape

    def test_period_call_finite(self):
        rope = self._make_period()
        d_head = DIM // NUM_HEADS
        x = jr.normal(KEY, (H * W, NUM_HEADS, d_head))
        out = rope(x, key=KEY, inference=True)
        assert jnp.all(jnp.isfinite(out))

    def test_period_call_changes_input(self):
        rope = self._make_period()
        d_head = DIM // NUM_HEADS
        x = jr.normal(KEY, (H * W, NUM_HEADS, d_head))
        out = rope(x, key=KEY, inference=True)
        assert not jnp.allclose(out, x)

    # -- strategy="period": determinism and augmentations --------------------

    def test_period_deterministic_in_inference(self):
        rope = self._make_period(
            shift_coords=0.1, jitter_coords=1.5, rescale_coords=1.5
        )
        sin1, cos1 = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(1), inference=True)
        sin2, cos2 = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(2), inference=True)
        assert jnp.allclose(sin1, sin2)
        assert jnp.allclose(cos1, cos2)

    def test_period_shift_stochastic_in_training(self):
        rope = self._make_period(shift_coords=0.5)
        sin1, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(1), inference=False)
        sin2, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(2), inference=False)
        assert not jnp.allclose(sin1, sin2)

    def test_period_jitter_stochastic_in_training(self):
        rope = self._make_period(jitter_coords=2.0)
        sin1, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(10), inference=False)
        sin2, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(20), inference=False)
        assert not jnp.allclose(sin1, sin2)

    def test_period_rescale_stochastic_in_training(self):
        rope = self._make_period(rescale_coords=2.0)
        sin1, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(10), inference=False)
        sin2, _ = rope.get_sincos(H=H, W=W, key=jr.PRNGKey(20), inference=False)
        assert not jnp.allclose(sin1, sin2)

    def test_period_training_requires_key(self):
        rope = self._make_period()
        with pytest.raises(ValueError, match="PRNG key is required"):
            rope.get_sincos(H=H, W=W, key=None, inference=False)

    # -- strategy="period": normalize_coords modes ---------------------------

    @pytest.mark.parametrize("normalize_coords", ["min", "max", "separate"])
    def test_period_normalize_coords_modes(self, normalize_coords):
        rope = self._make_period(normalize_coords=normalize_coords)
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.all(jnp.isfinite(sin))
        assert jnp.all(jnp.isfinite(cos))

    # -- strategy="period": frequency variants --------------------------------

    def test_period_min_max_period_mode(self):
        rope = VisionRoPE(
            strategy="period",
            dim=DIM,
            num_heads=NUM_HEADS,
            base=None,
            min_period=1.0,
            max_period=100.0,
        )
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.all(jnp.isfinite(sin))

    def test_period_freqs_shape(self):
        rope = self._make_period()
        d_quarter = (DIM // NUM_HEADS) // 4
        assert rope.freqs.shape == (d_quarter,)

    # -- strategy="mode": shapes and values ----------------------------------

    def test_mode_sincos_shape(self):
        dim = 32
        rope = self._make_mode(dim=dim)
        sin, cos = rope.get_sincos(H=H, W=W)
        expected_d = 2 * dim  # height + width concatenated
        assert sin.shape == (H * W, expected_d)
        assert cos.shape == (H * W, expected_d)

    def test_mode_sincos_finite(self):
        rope = self._make_mode()
        sin, cos = rope.get_sincos(H=H, W=W)
        assert jnp.all(jnp.isfinite(sin))
        assert jnp.all(jnp.isfinite(cos))

    def test_mode_sincos_bounded(self):
        rope = self._make_mode()
        sin, cos = rope.get_sincos(H=H, W=W)
        assert jnp.all(jnp.abs(sin) <= 1.0 + 1e-5)
        assert jnp.all(jnp.abs(cos) <= 1.0 + 1e-5)

    def test_mode_call_shape(self):
        dim = 32
        rope = self._make_mode(dim=dim)
        d_out = 2 * dim
        x = jr.normal(KEY, (H * W, NUM_HEADS, d_out))
        out = rope(x)
        assert out.shape == x.shape

    def test_mode_call_finite(self):
        dim = 32
        rope = self._make_mode(dim=dim)
        d_out = 2 * dim
        x = jr.normal(KEY, (H * W, NUM_HEADS, d_out))
        out = rope(x)
        assert jnp.all(jnp.isfinite(out))

    def test_mode_call_changes_input(self):
        dim = 32
        rope = self._make_mode(dim=dim)
        d_out = 2 * dim
        x = jr.normal(KEY, (H * W, NUM_HEADS, d_out))
        assert not jnp.allclose(rope(x), x)

    def test_mode_deterministic_no_key(self):
        rope = self._make_mode()
        sin1, cos1 = rope.get_sincos(H=H, W=W)
        sin2, cos2 = rope.get_sincos(H=H, W=W)
        assert jnp.allclose(sin1, sin2)
        assert jnp.allclose(cos1, cos2)

    # -- strategy="mode": frequency modes ------------------------------------

    @pytest.mark.parametrize("freqs_for", ["lang", "pixel", "constant"])
    def test_mode_freqs_for_variants(self, freqs_for):
        kwargs = dict(strategy="mode", freqs_for=freqs_for)
        if freqs_for == "constant":
            kwargs["num_freqs"] = 4
        else:
            kwargs["dim"] = DIM // 2
        rope = VisionRoPE(**kwargs)
        sin, cos = rope.get_sincos(H=H, W=W)
        assert jnp.all(jnp.isfinite(sin))

    def test_mode_custom_freqs(self):
        custom = jnp.array([1.0, 2.0, 3.0, 4.0])
        rope = VisionRoPE(strategy="mode", custom_freqs=custom)
        sin, cos = rope.get_sincos(H=H, W=W)
        assert sin.shape == (H * W, 2 * len(custom) * 2)  # repeat(2) per axis, concat
        assert jnp.all(jnp.isfinite(sin))

    def test_mode_pt_seq_len_scaling(self):
        """Different pt_seq_len should produce different embeddings."""
        rope_a = self._make_mode(pt_seq_len=14)
        rope_b = self._make_mode(pt_seq_len=28)
        sin_a, _ = rope_a.get_sincos(H=H, W=W)
        sin_b, _ = rope_b.get_sincos(H=H, W=W)
        assert not jnp.allclose(sin_a, sin_b)

    def test_mode_freqs_shape_lang(self):
        dim = 32
        rope = self._make_mode(dim=dim, freqs_for="lang")
        assert rope.freqs.shape == (dim // 2,)

    def test_mode_freqs_shape_pixel(self):
        dim = 32
        rope = self._make_mode(dim=dim, freqs_for="pixel")
        assert rope.freqs.shape == (dim // 2,)

    def test_mode_freqs_shape_constant(self):
        rope = VisionRoPE(strategy="mode", freqs_for="constant", num_freqs=7)
        assert rope.freqs.shape == (7,)

    # -- validation errors ---------------------------------------------------

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            VisionRoPE(strategy="unknown", dim=DIM)

    def test_period_missing_dim_raises(self):
        with pytest.raises(ValueError, match="requires `dim` and `num_heads`"):
            VisionRoPE(strategy="period", num_heads=NUM_HEADS)

    def test_period_missing_num_heads_raises(self):
        with pytest.raises(ValueError, match="requires `dim` and `num_heads`"):
            VisionRoPE(strategy="period", dim=DIM)

    def test_period_dim_not_divisible_raises(self):
        with pytest.raises(ValueError, match="4 \\* num_heads"):
            VisionRoPE(strategy="period", dim=65, num_heads=4)

    def test_period_base_and_periods_raises(self):
        with pytest.raises(ValueError, match="Exactly one of"):
            VisionRoPE(
                strategy="period",
                dim=DIM,
                num_heads=NUM_HEADS,
                base=100.0,
                min_period=1.0,
                max_period=10.0,
            )

    def test_period_neither_base_nor_periods_raises(self):
        with pytest.raises(ValueError, match="Exactly one of"):
            VisionRoPE(strategy="period", dim=DIM, num_heads=NUM_HEADS, base=None)

    def test_mode_missing_dim_lang_raises(self):
        with pytest.raises(ValueError, match="requires `dim`"):
            VisionRoPE(strategy="mode", freqs_for="lang")

    def test_mode_missing_dim_pixel_raises(self):
        with pytest.raises(ValueError, match="requires `dim`"):
            VisionRoPE(strategy="mode", freqs_for="pixel")

    def test_mode_unknown_modality_raises(self):
        with pytest.raises(ValueError, match="Unknown modality"):
            VisionRoPE(strategy="mode", dim=DIM, freqs_for="invalid")

    # -- parity with DinoRoPE -----------------------------------------------

    def test_period_parity_with_dinorope_inference(self):
        """VisionRoPE(strategy='period') must match DinoRoPE for the same config."""
        dino = DinoRoPE(DIM, num_heads=NUM_HEADS, base=100.0)
        vision = self._make_period(base=100.0)

        sin_d, cos_d = dino.get_sincos(H=H, W=W, key=KEY, inference=True)
        sin_v, cos_v = vision.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.allclose(sin_d, sin_v, atol=1e-6)
        assert jnp.allclose(cos_d, cos_v, atol=1e-6)

    def test_period_parity_with_dinorope_training(self):
        """Augmented outputs must also match for the same key."""
        cfg = dict(shift_coords=0.5, jitter_coords=2.0, rescale_coords=1.5)
        dino = DinoRoPE(DIM, num_heads=NUM_HEADS, **cfg)
        vision = self._make_period(**cfg)

        k = jr.PRNGKey(42)
        sin_d, cos_d = dino.get_sincos(H=H, W=W, key=k, inference=False)
        sin_v, cos_v = vision.get_sincos(H=H, W=W, key=k, inference=False)
        assert jnp.allclose(sin_d, sin_v, atol=1e-6)
        assert jnp.allclose(cos_d, cos_v, atol=1e-6)

    def test_period_parity_min_max_periods(self):
        cfg = dict(base=None, min_period=1.0, max_period=100.0)
        dino = DinoRoPE(DIM, num_heads=NUM_HEADS, **cfg)
        vision = VisionRoPE(strategy="period", dim=DIM, num_heads=NUM_HEADS, **cfg)

        sin_d, cos_d = dino.get_sincos(H=H, W=W, key=KEY, inference=True)
        sin_v, cos_v = vision.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.allclose(sin_d, sin_v, atol=1e-6)
        assert jnp.allclose(cos_d, cos_v, atol=1e-6)

    # -- registry integration ------------------------------------------------

    def test_registry_lookup(self):
        assert get_posemb("visionrope") is VisionRoPE

    def test_registry_roundtrip_period(self):
        cls = get_posemb("visionrope")
        rope = cls(strategy="period", dim=DIM, num_heads=NUM_HEADS)
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert jnp.all(jnp.isfinite(sin))

    def test_registry_roundtrip_mode(self):
        cls = get_posemb("visionrope")
        rope = cls(strategy="mode", dim=DIM // 2)
        sin, cos = rope.get_sincos(H=H, W=W)
        assert jnp.all(jnp.isfinite(sin))

    # -- dtype handling ------------------------------------------------------

    def test_period_dtype_propagated(self):
        rope = self._make_period(dtype=jnp.bfloat16)
        sin, cos = rope.get_sincos(H=H, W=W, key=KEY, inference=True)
        assert sin.dtype == jnp.bfloat16
        assert cos.dtype == jnp.bfloat16

    def test_mode_dtype_propagated(self):
        rope = self._make_mode(dtype=jnp.bfloat16)
        sin, cos = rope.get_sincos(H=H, W=W)
        assert sin.dtype == jnp.bfloat16
        assert cos.dtype == jnp.bfloat16

    # -- non-square grids (mode strategy) ------------------------------------

    def test_mode_nonsquare_get_sincos(self):
        rope = self._make_mode()
        sin, cos = rope.get_sincos(H=4, W=16)
        assert sin.shape[0] == 4 * 16
        assert jnp.all(jnp.isfinite(sin))

    @pytest.mark.parametrize("normalize_coords", ["min", "max", "separate"])
    def test_period_nonsquare_get_sincos(self, normalize_coords):
        rope = self._make_period(normalize_coords=normalize_coords)
        sin, cos = rope.get_sincos(H=4, W=16, key=KEY, inference=True)
        assert sin.shape[0] == 4 * 16
        assert jnp.all(jnp.isfinite(sin))


# PosCNN


class TestPosCNN:
    def test_output_shape_stride1(self):
        layer = PosCNN(DIM, DIM, key=KEY, s=1)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert layer(x).shape == (SEQLEN, DIM)

    def test_output_finite(self):
        layer = PosCNN(DIM, DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_residual_when_stride1(self):
        """stride=1 adds the projection to the input (residual)."""
        layer = PosCNN(DIM, DIM, key=KEY, s=1)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out = layer(x)
        # Output must differ from input (proj != zero) but shape is preserved
        assert out.shape == x.shape

    def test_no_residual_when_stride2(self):
        """stride > 1 returns only the projection (no residual)."""
        out_channels = DIM
        layer = PosCNN(DIM, out_channels, key=KEY, s=2)
        x = jr.normal(KEY, (SEQLEN, DIM))
        # With stride=2, spatial dims halve: H/2 * W/2 = 16 tokens
        out = layer(x)
        assert out.shape == (SEQLEN // 4, out_channels)

    def test_non_square_seqlen_raises(self):
        layer = PosCNN(DIM, DIM, key=KEY)
        x = jr.normal(KEY, (15, DIM))  # 15 is not a perfect square
        with pytest.raises(ValueError, match="perfect square"):
            layer(x)

    def test_bfloat16_input_finite(self):
        layer = PosCNN(DIM, DIM, key=KEY)
        dtype = jnp.bfloat16
        layer = jax.tree_util.tree_map(
            lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
            layer,
        )
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(dtype)
        assert jnp.all(jnp.isfinite(layer(x)))


# PosCNN2D


class TestPosCNN2D:
    def test_output_shape(self):
        layer = PosCNN2D(DIM, key=KEY)
        x = jr.normal(KEY, (DIM, H, W))
        assert layer(x, key=KEY).shape == (DIM, H, W)

    def test_output_finite(self):
        layer = PosCNN2D(DIM, key=KEY)
        x = jr.normal(KEY, (DIM, H, W))
        assert jnp.all(jnp.isfinite(layer(x, key=KEY)))

    def test_residual_enabled_by_default_when_same_channels(self):
        """residual=True when stride=1 and out_channels defaults to in_channels."""
        layer = PosCNN2D(DIM, key=KEY)
        assert layer.residual is True

    def test_residual_disabled_when_out_channels_differ(self):
        """residual=False when in_channels != out_channels."""
        layer = PosCNN2D(DIM, out_channels=DIM // 2, key=KEY)
        assert layer.residual is False

    def test_no_norm_layer(self):
        layer = PosCNN2D(DIM, norm_layer=None, key=KEY)
        x = jr.normal(KEY, (DIM, H, W))
        assert jnp.all(jnp.isfinite(layer(x, key=KEY)))

    def test_deterministic_in_inference(self):
        layer = PosCNN2D(DIM, key=KEY)
        x = jr.normal(KEY, (DIM, H, W))
        out1 = layer(x, key=jr.PRNGKey(1), inference=True)
        out2 = layer(x, key=jr.PRNGKey(2), inference=True)
        assert jnp.allclose(out1, out2)

    def test_output_changes_input(self):
        """Adding the conv encoding changes the spatial features."""
        layer = PosCNN2D(DIM, key=KEY)
        x = jr.normal(KEY, (DIM, H, W))
        out = layer(x, key=KEY, inference=True)
        assert not jnp.allclose(out, x)


# get_posemb


class TestGetPosemb:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("learnedposembed", LearnedPosEmbed),
            ("posembmlpswinv1d", PosEmbMLPSwinv1D),
            ("posembmlpswinv2d", PosEmbMLPSwinv2D),
            ("rope", RoPE),
            ("dinorope", DinoRoPE),
            ("poscnn", PosCNN),
            ("poscnn2d", PosCNN2D),
        ],
    )
    def test_string_resolution(self, name, expected):
        assert get_posemb(name) is expected

    def test_class_passthrough(self):
        assert get_posemb(RoPE) is RoPE

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown module string"):
            get_posemb("nonexistent_posemb")


# register_posemb


class TestRegisterPosemb:
    def test_register_default_name(self):
        from equimo.layers.posemb import _POSEMB_REGISTRY

        @register_posemb()
        class CustomPosEmb(eqx.Module):
            pass

        assert "customposemb" in _POSEMB_REGISTRY
        assert get_posemb("customposemb") is CustomPosEmb

    def test_register_custom_name(self):
        from equimo.layers.posemb import _POSEMB_REGISTRY

        @register_posemb(name="MySpecialPosEmb")
        class CustomPosEmb2(eqx.Module):
            pass

        assert "myspecialposemb" in _POSEMB_REGISTRY
        assert get_posemb("myspecialposemb") is CustomPosEmb2

    def test_register_non_eqx_module_raises(self):
        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):

            @register_posemb()
            class NotAModule:
                pass

    def test_register_duplicate_name_raises(self):
        @register_posemb()
        class DuplicatePosEmb(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_posemb(name="DuplicatePosEmb")
            class AnotherPosEmb(eqx.Module):
                pass
