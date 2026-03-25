"""Tests for equimo.layers.convolution."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import equinox as eqx

from equimo.layers.convolution import (
    SingleConvBlock,
    DoubleConvBlock,
    Stem,
    ConvBottleneck,
    C2f,
    C3k,
    C3,
    C3k2,
    MBConv,
    DSConv,
    UIB,
    IFormerStem,
    IFormerBlock,
    GenericGhostModule,
    GhostBottleneck,
    PartialConv2d,
    FasterNetBlock,
    GLUConv,
    ATConv,
    ATConvBlock,
    S2Mixer,
    ShiftNeck,
    ShiftFFN,
    FreeNetBlock,
    get_conv,
    register_conv,
)

KEY = jr.PRNGKey(0)
IN_CHANNELS = 16
OUT_CHANNELS = 32
H, W = 16, 16


class TestConvolutionLayers:
    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (
                SingleConvBlock,
                {"in_channels": IN_CHANNELS, "out_channels": OUT_CHANNELS},
            ),
            (DoubleConvBlock, {"dim": IN_CHANNELS}),
            (
                Stem,
                {
                    "in_channels": 3,
                    "embed_dim": IN_CHANNELS,
                    "img_size": 32,
                    "patch_size": 4,
                },
            ),
            (ConvBottleneck, {"in_channels": IN_CHANNELS, "out_channels": IN_CHANNELS}),
            (C2f, {"in_channels": IN_CHANNELS, "out_channels": IN_CHANNELS}),
            (C3k, {"in_channels": IN_CHANNELS, "out_channels": IN_CHANNELS}),
            (C3, {"in_channels": IN_CHANNELS, "out_channels": IN_CHANNELS}),
            (C3k2, {"in_channels": IN_CHANNELS, "out_channels": IN_CHANNELS}),
            (MBConv, {"in_channels": IN_CHANNELS, "out_channels": OUT_CHANNELS}),
            (DSConv, {"in_channels": IN_CHANNELS, "out_channels": OUT_CHANNELS}),
            (
                UIB,
                {
                    "in_channels": IN_CHANNELS,
                    "out_channels": OUT_CHANNELS,
                    "start_dw_kernel_size": 3,
                    "middle_dw_kernel_size": 3,
                },
            ),
            (IFormerStem, {"in_channels": 3, "out_channels": IN_CHANNELS}),
            (IFormerBlock, {"dim": IN_CHANNELS}),
            (
                GenericGhostModule,
                {"in_channels": IN_CHANNELS, "out_channels": OUT_CHANNELS},
            ),
            (
                GhostBottleneck,
                {
                    "in_channels": IN_CHANNELS,
                    "mid_channels": IN_CHANNELS * 2,
                    "out_channels": OUT_CHANNELS,
                },
            ),
            (PartialConv2d, {"in_channels": IN_CHANNELS, "n_dim": 4}),
            (FasterNetBlock, {"dim": IN_CHANNELS}),
            (GLUConv, {"in_channels": IN_CHANNELS, "hidden_channels": IN_CHANNELS * 2}),
            (ATConv, {"in_channels": IN_CHANNELS}),
            (ATConvBlock, {"dim": IN_CHANNELS}),
            (S2Mixer, {"in_channels": IN_CHANNELS}),
            (ShiftNeck, {"in_channels": IN_CHANNELS}),
            (ShiftFFN, {"in_channels": IN_CHANNELS}),
            (FreeNetBlock, {"dim": IN_CHANNELS}),
        ],
    )
    def test_output_shape_and_finite(self, cls, kwargs):
        key = KEY
        model = cls(**kwargs, key=key)

        # Determine input shape
        if cls in (Stem, IFormerStem):
            x = jr.normal(key, (3, 32, 32))
        elif cls == PartialConv2d:
            x = jr.normal(key, (IN_CHANNELS, H, W))
        else:
            in_c = kwargs.get("in_channels") or kwargs.get("dim") or IN_CHANNELS
            x = jr.normal(key, (in_c, H, W))

        out = model(x, key=key, inference=True)

        if cls == Stem:
            # Stem flattens to (seqlen, dim)
            expected_patches = (32 // 4) ** 2
            assert out.shape == (expected_patches, IN_CHANNELS)
        elif cls == ShiftNeck:
            assert out.shape == x.shape
        else:
            # For most others, spatial dims might change depending on stride
            pass

        assert jnp.all(jnp.isfinite(out))

    def test_registry(self):
        assert get_conv("singleconvblock") is SingleConvBlock
        assert get_conv("doubleconvblock") is DoubleConvBlock
        assert get_conv("stem") is Stem

    def test_low_precision(self):
        model = SingleConvBlock(IN_CHANNELS, OUT_CHANNELS, key=KEY)
        model = jax.tree_util.tree_map(
            lambda leaf: (
                leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf
            ),
            model,
        )
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(jnp.bfloat16)
        out = model(x, inference=True)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))

    def test_ghostnet_fusion(self):
        from equimo.layers.convolution import update_ghostnet, finalize_ghostnet

        model = GhostBottleneck(IN_CHANNELS, IN_CHANNELS * 2, OUT_CHANNELS, key=KEY)

        # Test update_ghostnet
        fused_model = update_ghostnet(model)
        assert isinstance(fused_model, GhostBottleneck)

        # Test finalize_ghostnet
        final_model = finalize_ghostnet(model)
        assert final_model.inference is True

        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        out = final_model(x, inference=True)
        assert out.shape[0] == OUT_CHANNELS
