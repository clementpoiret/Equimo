"""Tests for equimo.layers.patch."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.patch import (
    ConvPatchEmbed,
    PatchEmbedding,
    PatchMerging,
    SEPatchMerging,
    _PATCH_REGISTRY,
    get_patch,
    register_patch,
)

KEY = jr.PRNGKey(0)


# ---------------------------------------------------------------------------
# PatchEmbedding
# ---------------------------------------------------------------------------


class TestPatchEmbedding:
    IN_CHANNELS = 3
    DIM = 96
    PATCH_SIZE = 16
    IMG_SIZE = 224

    def test_output_shape_flattened(self):
        """flatten=True → (num_patches, dim)."""
        layer = PatchEmbedding(
            self.IN_CHANNELS, self.DIM, self.PATCH_SIZE, key=KEY, flatten=True
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.IMG_SIZE, self.IMG_SIZE))
        expected_patches = (self.IMG_SIZE // self.PATCH_SIZE) ** 2
        assert layer(x).shape == (expected_patches, self.DIM)

    def test_output_shape_spatial(self):
        """flatten=False → (dim, grid_h, grid_w)."""
        layer = PatchEmbedding(
            self.IN_CHANNELS, self.DIM, self.PATCH_SIZE, key=KEY, flatten=False
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.IMG_SIZE, self.IMG_SIZE))
        grid = self.IMG_SIZE // self.PATCH_SIZE
        assert layer(x).shape == (self.DIM, grid, grid)

    def test_output_finite(self):
        layer = PatchEmbedding(self.IN_CHANNELS, self.DIM, self.PATCH_SIZE, key=KEY)
        x = jr.normal(KEY, (self.IN_CHANNELS, self.IMG_SIZE, self.IMG_SIZE))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_no_norm_layer_default(self):
        """norm_layer=None (default) must produce a valid output without error."""
        layer = PatchEmbedding(self.IN_CHANNELS, self.DIM, self.PATCH_SIZE, key=KEY)
        x = jr.normal(KEY, (self.IN_CHANNELS, self.IMG_SIZE, self.IMG_SIZE))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_with_norm_layer(self):
        layer = PatchEmbedding(
            self.IN_CHANNELS,
            self.DIM,
            self.PATCH_SIZE,
            norm_layer=eqx.nn.LayerNorm,
            key=KEY,
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.IMG_SIZE, self.IMG_SIZE))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_num_patches_metadata(self):
        layer = PatchEmbedding(
            self.IN_CHANNELS,
            self.DIM,
            self.PATCH_SIZE,
            img_size=self.IMG_SIZE,
            key=KEY,
        )
        expected = (self.IMG_SIZE // self.PATCH_SIZE) ** 2
        assert layer.num_patches == expected
        assert layer.grid_size == (
            self.IMG_SIZE // self.PATCH_SIZE,
            self.IMG_SIZE // self.PATCH_SIZE,
        )

    def test_no_img_size_metadata_is_none(self):
        layer = PatchEmbedding(self.IN_CHANNELS, self.DIM, self.PATCH_SIZE, key=KEY)
        assert layer.img_size is None
        assert layer.grid_size is None
        assert layer.num_patches is None

    def test_wrong_height_raises(self):
        layer = PatchEmbedding(
            self.IN_CHANNELS, self.DIM, self.PATCH_SIZE, img_size=self.IMG_SIZE, key=KEY
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, 192, self.IMG_SIZE))
        with pytest.raises(AssertionError, match="height"):
            layer(x)

    def test_wrong_width_raises(self):
        layer = PatchEmbedding(
            self.IN_CHANNELS, self.DIM, self.PATCH_SIZE, img_size=self.IMG_SIZE, key=KEY
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.IMG_SIZE, 192))
        with pytest.raises(AssertionError, match="width"):
            layer(x)

    def test_dynamic_img_size(self):
        """dynamic_img_size=True allows different (divisible) input sizes."""
        layer = PatchEmbedding(
            self.IN_CHANNELS,
            self.DIM,
            self.PATCH_SIZE,
            img_size=self.IMG_SIZE,
            dynamic_img_size=True,
            key=KEY,
        )
        alt_size = 192
        x = jr.normal(KEY, (self.IN_CHANNELS, alt_size, alt_size))
        out = layer(x)
        assert out.shape == ((alt_size // self.PATCH_SIZE) ** 2, self.DIM)

    def test_dynamic_img_pad_non_divisible(self):
        """dynamic_img_pad=True pads non-divisible sizes transparently."""
        layer = PatchEmbedding(
            self.IN_CHANNELS,
            self.DIM,
            self.PATCH_SIZE,
            dynamic_img_size=True,
            dynamic_img_pad=True,
            key=KEY,
        )
        non_div_size = 220
        x = jr.normal(KEY, (self.IN_CHANNELS, non_div_size, non_div_size))
        out = layer(x)
        expected_patches = math.ceil(non_div_size / self.PATCH_SIZE) ** 2
        assert out.shape == (expected_patches, self.DIM)
        assert jnp.all(jnp.isfinite(out))

    def test_rectangular_patch_size(self):
        """Tuple patch sizes produce the correct grid."""
        layer = PatchEmbedding(self.IN_CHANNELS, self.DIM, (8, 16), key=KEY)
        x = jr.normal(KEY, (self.IN_CHANNELS, 128, 128))
        # grid: 128//8 × 128//16 = 16 × 8 = 128 patches
        assert layer(x).shape == (128, self.DIM)

    def test_dtype_preserved_bfloat16(self):
        layer = PatchEmbedding(self.IN_CHANNELS, self.DIM, self.PATCH_SIZE, key=KEY)
        x = jr.normal(KEY, (self.IN_CHANNELS, self.IMG_SIZE, self.IMG_SIZE)).astype(
            jnp.bfloat16
        )
        out = layer(x)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_float16(self):
        layer = PatchEmbedding(self.IN_CHANNELS, self.DIM, self.PATCH_SIZE, key=KEY)
        x = jr.normal(KEY, (self.IN_CHANNELS, self.IMG_SIZE, self.IMG_SIZE)).astype(
            jnp.float16
        )
        out = layer(x)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_with_norm_layer(self):
        """dtype preservation must hold even when a norm layer is applied."""
        layer = PatchEmbedding(
            self.IN_CHANNELS,
            self.DIM,
            self.PATCH_SIZE,
            norm_layer=eqx.nn.LayerNorm,
            key=KEY,
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.IMG_SIZE, self.IMG_SIZE)).astype(
            jnp.bfloat16
        )
        out = layer(x)
        assert out.dtype == jnp.bfloat16


# ---------------------------------------------------------------------------
# ConvPatchEmbed
# ---------------------------------------------------------------------------


class TestConvPatchEmbed:
    IN_CHANNELS = 3
    HIDDEN_CHANNELS = 64
    OUT_CHANNELS = 128
    H, W = 64, 64

    def test_output_shape(self):
        """Two stride-2 convs reduce H and W by 4× each."""
        layer = ConvPatchEmbed(
            self.IN_CHANNELS, self.HIDDEN_CHANNELS, self.OUT_CHANNELS, key=KEY
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W))
        assert layer(x).shape == (self.OUT_CHANNELS, self.H // 4, self.W // 4)

    def test_output_finite(self):
        layer = ConvPatchEmbed(
            self.IN_CHANNELS, self.HIDDEN_CHANNELS, self.OUT_CHANNELS, key=KEY
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_custom_act_layer(self):
        layer = ConvPatchEmbed(
            self.IN_CHANNELS,
            self.HIDDEN_CHANNELS,
            self.OUT_CHANNELS,
            act_layer=jax.nn.gelu,
            key=KEY,
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_act_is_static_field(self):
        """act must be stored as configured and not appear as a dynamic array leaf."""
        layer = ConvPatchEmbed(
            self.IN_CHANNELS,
            self.HIDDEN_CHANNELS,
            self.OUT_CHANNELS,
            act_layer=jax.nn.gelu,
            key=KEY,
        )
        assert layer.act is jax.nn.gelu
        # Static fields are not pytree leaves; act must not appear in tree_leaves
        leaves = jax.tree_util.tree_leaves(layer)
        assert layer.act not in leaves

    def test_arbitrary_channel_sizes(self):
        layer = ConvPatchEmbed(1, 32, 256, key=KEY)
        x = jr.normal(KEY, (1, self.H, self.W))
        assert layer(x).shape == (256, self.H // 4, self.W // 4)

    def test_dtype_preserved_bfloat16(self):
        layer = ConvPatchEmbed(
            self.IN_CHANNELS, self.HIDDEN_CHANNELS, self.OUT_CHANNELS, key=KEY
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W)).astype(jnp.bfloat16)
        out = layer(x)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_float16(self):
        layer = ConvPatchEmbed(
            self.IN_CHANNELS, self.HIDDEN_CHANNELS, self.OUT_CHANNELS, key=KEY
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W)).astype(jnp.float16)
        out = layer(x)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# PatchMerging
# ---------------------------------------------------------------------------


class TestPatchMerging:
    DIM = 96
    SEQLEN = 196  # 14×14 grid

    def test_default_output_shape_doubles_channels(self):
        """Default: output channels = 2 × input dim."""
        layer = PatchMerging(self.DIM, key=KEY)
        x = jr.normal(KEY, (self.SEQLEN, self.DIM))
        assert layer(x).shape == (self.SEQLEN // 4, self.DIM * 2)

    def test_custom_out_dim(self):
        """out_dim overrides the default channel doubling."""
        out_dim = 48
        layer = PatchMerging(self.DIM, out_dim=out_dim, key=KEY)
        x = jr.normal(KEY, (self.SEQLEN, self.DIM))
        assert layer(x).shape == (self.SEQLEN // 4, out_dim)

    def test_output_finite(self):
        layer = PatchMerging(self.DIM, key=KEY)
        x = jr.normal(KEY, (self.SEQLEN, self.DIM))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_custom_ratio(self):
        """Custom expansion ratio must not affect the output shape."""
        layer = PatchMerging(self.DIM, ratio=2.0, key=KEY)
        x = jr.normal(KEY, (self.SEQLEN, self.DIM))
        assert layer(x).shape == (self.SEQLEN // 4, self.DIM * 2)

    def test_larger_grid(self):
        layer = PatchMerging(self.DIM, key=KEY)
        x = jr.normal(KEY, (784, self.DIM))  # 28×28
        assert layer(x).shape == (196, self.DIM * 2)

    def test_conv_types_are_single_conv_block(self):
        """conv1/2/3 must be SingleConvBlock instances."""
        from equimo.layers.convolution import SingleConvBlock

        layer = PatchMerging(self.DIM, key=KEY)
        assert isinstance(layer.conv1, SingleConvBlock)
        assert isinstance(layer.conv2, SingleConvBlock)
        assert isinstance(layer.conv3, SingleConvBlock)

    def test_dtype_preserved_bfloat16(self):
        layer = PatchMerging(self.DIM, key=KEY)
        x = jr.normal(KEY, (self.SEQLEN, self.DIM)).astype(jnp.bfloat16)
        out = layer(x)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_float16(self):
        layer = PatchMerging(self.DIM, key=KEY)
        x = jr.normal(KEY, (self.SEQLEN, self.DIM)).astype(jnp.float16)
        out = layer(x)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# SEPatchMerging
# ---------------------------------------------------------------------------


class TestSEPatchMerging:
    IN_CHANNELS = 64
    OUT_CHANNELS = 128
    H, W = 32, 32

    def test_output_shape(self):
        """Stride-2 conv halves spatial dims."""
        layer = SEPatchMerging(self.IN_CHANNELS, self.OUT_CHANNELS, key=KEY)
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W))
        assert layer(x).shape == (self.OUT_CHANNELS, self.H // 2, self.W // 2)

    def test_output_finite(self):
        layer = SEPatchMerging(self.IN_CHANNELS, self.OUT_CHANNELS, key=KEY)
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_same_in_out_channels(self):
        layer = SEPatchMerging(self.IN_CHANNELS, self.IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W))
        assert layer(x).shape == (self.IN_CHANNELS, self.H // 2, self.W // 2)

    def test_custom_expansion_ratio(self):
        """expansion_ratio controls the hidden channel count."""
        layer = SEPatchMerging(
            self.IN_CHANNELS, self.OUT_CHANNELS, expansion_ratio=2.0, key=KEY
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W))
        out = layer(x)
        assert out.shape == (self.OUT_CHANNELS, self.H // 2, self.W // 2)
        assert jnp.all(jnp.isfinite(out))

    def test_custom_act_layer(self):
        layer = SEPatchMerging(
            self.IN_CHANNELS, self.OUT_CHANNELS, act_layer=jax.nn.gelu, key=KEY
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_act_is_static_field(self):
        """act must be stored as configured and not appear as a dynamic array leaf."""
        layer = SEPatchMerging(
            self.IN_CHANNELS, self.OUT_CHANNELS, act_layer=jax.nn.gelu, key=KEY
        )
        assert layer.act is jax.nn.gelu
        leaves = jax.tree_util.tree_leaves(layer)
        assert layer.act not in leaves

    def test_custom_norm_max_group(self):
        layer = SEPatchMerging(
            self.IN_CHANNELS, self.OUT_CHANNELS, norm_max_group=16, key=KEY
        )
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_dtype_preserved_bfloat16(self):
        layer = SEPatchMerging(self.IN_CHANNELS, self.OUT_CHANNELS, key=KEY)
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W)).astype(jnp.bfloat16)
        out = layer(x)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_dtype_preserved_float16(self):
        layer = SEPatchMerging(self.IN_CHANNELS, self.OUT_CHANNELS, key=KEY)
        x = jr.normal(KEY, (self.IN_CHANNELS, self.H, self.W)).astype(jnp.float16)
        out = layer(x)
        assert out.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# get_patch
# ---------------------------------------------------------------------------


class TestGetPatch:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("patchembedding", PatchEmbedding),
            ("convpatchembed", ConvPatchEmbed),
            ("patchmerging", PatchMerging),
            ("sepatchmerging", SEPatchMerging),
        ],
    )
    def test_string_resolution(self, name, expected):
        assert get_patch(name) is expected

    def test_class_passthrough(self):
        assert get_patch(PatchEmbedding) is PatchEmbedding

    def test_class_passthrough_arbitrary(self):
        assert get_patch(PatchMerging) is PatchMerging

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown module string"):
            get_patch("nonexistent_patch")

    def test_returned_class_is_instantiable_patchmerging(self):
        cls = get_patch("patchmerging")
        layer = cls(64, key=KEY)
        x = jr.normal(KEY, (196, 64))
        assert layer(x).shape == (49, 128)


# ---------------------------------------------------------------------------
# register_patch
# ---------------------------------------------------------------------------


class TestRegisterPatch:
    def test_register_default_name(self):
        @register_patch()
        class MyPatchModule(eqx.Module):
            pass

        assert "mypatchmodule" in _PATCH_REGISTRY
        assert get_patch("mypatchmodule") is MyPatchModule

    def test_register_custom_name(self):
        @register_patch(name="SuperPatch99")
        class AnotherPatch(eqx.Module):
            pass

        assert "superpatch99" in _PATCH_REGISTRY
        assert get_patch("superpatch99") is AnotherPatch

    def test_register_non_eqx_module_raises(self):
        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):

            @register_patch()
            class NotAModule:
                pass

    def test_register_duplicate_name_raises(self):
        @register_patch()
        class UniquePatch(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_patch(name="UniquePatch")
            class AnotherOne(eqx.Module):
                pass
