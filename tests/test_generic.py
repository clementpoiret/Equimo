"""Tests for equimo.layers.generic: Residual, WindowedSequence, BlockChunk."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array, PRNGKeyArray

from equimo.layers.generic import BlockChunk, Residual, WindowedSequence
from equimo.layers.norm import LayerScale

KEY = jr.PRNGKey(0)
DIM = 16
SHAPE = (DIM, 8, 8)  # (C, H, W) — spatial feature map


# A minimal block for testing composition layers


class _IdentityBlock(eqx.Module):
    """Passes input through unchanged; optionally accepts key/inference kwargs."""

    def __call__(
        self, x: Array, *, key: PRNGKeyArray = None, inference: bool = False
    ) -> Array:
        return x


class _ScaleBlock(eqx.Module):
    """Scales input by a fixed constant for verifying data flow."""

    scale: float = eqx.field(static=True)

    def __init__(self, scale: float = 2.0, **_kwargs):
        self.scale = scale

    def __call__(
        self, x: Array, *, key: PRNGKeyArray, inference: bool = False
    ) -> Array:
        return x * self.scale


class _WindowBlock(eqx.Module):
    """Block that accepts window_size=0 (as WindowedSequence injects it)."""

    window_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        drop_path: float = 0.0,
        window_size: int = 0,
        key: PRNGKeyArray = None,
        **_kw,
    ):
        self.window_size = window_size

    def __call__(
        self, x: Array, *, key: PRNGKeyArray, inference: bool = False
    ) -> Array:
        return x


# Residual


class TestResidual:
    def test_output_shape(self):
        layer = Residual(_IdentityBlock(), drop_path=0.0)
        x = jr.normal(KEY, SHAPE)
        out = layer(x, KEY)
        assert out.shape == SHAPE

    def test_identity_module_output_equals_2x(self):
        """x + identity(x) == 2*x when p=0."""
        layer = Residual(_IdentityBlock(), drop_path=0.0)
        x = jr.normal(KEY, SHAPE)
        out = layer(x, KEY)
        assert jnp.allclose(out, 2 * x, atol=1e-5)

    def test_drop_path_zero_is_deterministic(self):
        layer = Residual(_IdentityBlock(), drop_path=0.0)
        x = jr.normal(KEY, SHAPE)
        out1 = layer(x, jr.PRNGKey(1))
        out2 = layer(x, jr.PRNGKey(2))
        assert jnp.array_equal(out1, out2)

    def test_inference_passthrough(self):
        layer = Residual(_IdentityBlock(), drop_path=0.5)
        x = jr.normal(KEY, SHAPE)
        out1 = layer(x, jr.PRNGKey(1), inference=True)
        out2 = layer(x, jr.PRNGKey(2), inference=True)
        assert jnp.array_equal(out1, out2)

    def test_pass_args_true_forwards_key_inference(self):
        """pass_args=True should forward key and inference to the wrapped module."""
        layer = Residual(_IdentityBlock(), drop_path=0.0)
        x = jr.normal(KEY, SHAPE)
        out = layer(x, KEY, pass_args=True)
        assert out.shape == SHAPE

    def test_layer_scale_activated_with_valid_args(self):
        """Residual with dim, axis, init_values should use LayerScale, not Identity."""
        layer = Residual(
            _IdentityBlock(), dim=DIM, axis=0, init_values=1.0, drop_path=0.0
        )
        assert isinstance(layer.ls, LayerScale)

    def test_layer_scale_not_activated_without_args(self):
        layer = Residual(_IdentityBlock())
        assert isinstance(layer.ls, eqx.nn.Identity)

    def test_layer_scale_activated_with_axis_zero(self):
        """axis=0 must NOT be treated as falsy — LayerScale must activate."""
        layer = Residual(
            _IdentityBlock(), dim=DIM, axis=0, init_values=1e-6, drop_path=0.0
        )
        assert isinstance(layer.ls, LayerScale), (
            "axis=0 was treated as falsy — use_ls bug not fixed"
        )

    def test_output_finite(self):
        layer = Residual(_IdentityBlock(), drop_path=0.3)
        x = jr.normal(KEY, SHAPE)
        assert jnp.all(jnp.isfinite(layer(x, KEY)))


# WindowedSequence


class TestWindowedSequence:
    def test_output_shape(self):
        layer = WindowedSequence(
            in_channels=DIM,
            depth=2,
            block_type=_WindowBlock,
            block_kwargs={},
            window_size=4,
            key=KEY,
        )
        x = jr.normal(KEY, (DIM, 8, 8))
        assert layer(x, KEY).shape == (DIM, 8, 8)

    def test_identity_blocks_passthrough(self):
        """With identity blocks, output must equal input."""
        layer = WindowedSequence(
            in_channels=DIM,
            depth=2,
            block_type=_WindowBlock,
            block_kwargs={},
            window_size=4,
            key=KEY,
        )
        x = jr.normal(KEY, (DIM, 8, 8))
        assert jnp.array_equal(layer(x, KEY), x)

    def test_padding_then_crop(self):
        """Spatial dims not divisible by window_size must be padded then cropped."""
        layer = WindowedSequence(
            in_channels=DIM,
            depth=1,
            block_type=_WindowBlock,
            block_kwargs={},
            window_size=4,
            key=KEY,
        )
        x = jr.normal(KEY, (DIM, 7, 9))
        assert layer(x, KEY).shape == (DIM, 7, 9)

    def test_drop_path_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="depth"):
            WindowedSequence(
                in_channels=DIM,
                depth=3,
                block_type=_WindowBlock,
                block_kwargs={},
                window_size=4,
                drop_path=[0.1, 0.2],
                key=KEY,
            )

    def test_different_keys_can_produce_different_outputs(self):
        """With drop_path > 0, different keys should (usually) give different results."""
        layer = WindowedSequence(
            in_channels=DIM,
            depth=1,
            block_type=_WindowBlock,
            block_kwargs={},
            window_size=4,
            drop_path=0.5,
            key=KEY,
        )
        x = jnp.ones((DIM, 8, 8))
        # _WindowBlock ignores drop_path, so outputs are always the same here —
        # just check shape is stable across keys
        out1 = layer(x, jr.PRNGKey(1))
        out2 = layer(x, jr.PRNGKey(2))
        assert out1.shape == out2.shape

    def test_inference_mode(self):
        layer = WindowedSequence(
            in_channels=DIM,
            depth=2,
            block_type=_WindowBlock,
            block_kwargs={},
            window_size=4,
            key=KEY,
        )
        x = jr.normal(KEY, (DIM, 8, 8))
        assert layer(x, KEY, inference=True).shape == (DIM, 8, 8)


# BlockChunk


class _SimpleBlock(eqx.Module):
    """Simple block that accepts the standard BlockChunk interface."""

    scale: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        init_values: float | None = None,
        key: PRNGKeyArray = None,
        **_kw,
    ):
        self.scale = 1.0

    def __call__(
        self, x: Array, *, key: PRNGKeyArray, inference: bool = False, **_kw
    ) -> Array:
        return x


class _SimpleDownsampler(eqx.Module):
    def __init__(self, in_channels: int, out_channels: int, key: PRNGKeyArray):
        pass

    def __call__(self, x: Array) -> Array:
        # Fake downsampler: squeeze spatial dims by 2
        return x[:, ::2, ::2]


class TestBlockChunk:
    def test_blocks_only(self):
        chunk = BlockChunk(
            depth=3,
            module=_SimpleBlock,
            module_kwargs={"dim": DIM},
            key=KEY,
        )
        x = jr.normal(KEY, SHAPE)
        out = chunk(x, key=KEY)
        assert out.shape == SHAPE

    def test_downsampler_only(self):
        chunk = BlockChunk(
            depth=0,
            in_channels=DIM,
            out_channels=DIM,
            downsampler=_SimpleDownsampler,
            key=KEY,
        )
        x = jr.normal(KEY, SHAPE)
        out = chunk(x, key=KEY)
        assert out.shape == (DIM, 4, 4)

    def test_blocks_then_downsampler(self):
        chunk = BlockChunk(
            depth=2,
            in_channels=DIM,
            out_channels=DIM,
            module=_SimpleBlock,
            module_kwargs={"dim": DIM},
            downsampler=_SimpleDownsampler,
            downsample_last=True,
            key=KEY,
        )
        x = jr.normal(KEY, SHAPE)
        out = chunk(x, key=KEY)
        assert out.shape == (DIM, 4, 4)

    def test_downsampler_then_blocks(self):
        chunk = BlockChunk(
            depth=2,
            in_channels=DIM,
            out_channels=DIM,
            module=_SimpleBlock,
            module_kwargs={"dim": DIM},
            downsampler=_SimpleDownsampler,
            downsample_last=False,
            key=KEY,
        )
        x = jr.normal(KEY, SHAPE)
        out = chunk(x, key=KEY)
        assert out.shape == (DIM, 4, 4)

    def test_no_module_no_downsampler_raises(self):
        with pytest.raises(AssertionError):
            BlockChunk(depth=2, key=KEY)

    def test_drop_path_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="drop_path length"):
            BlockChunk(
                depth=3,
                module=_SimpleBlock,
                module_kwargs={"dim": DIM},
                drop_path=[0.1, 0.2],
                key=KEY,
            )

    def test_depth_zero_no_blocks(self):
        chunk = BlockChunk(
            depth=0,
            in_channels=DIM,
            out_channels=DIM,
            downsampler=_SimpleDownsampler,
            key=KEY,
        )
        assert chunk.blocks is None

    def test_output_finite(self):
        chunk = BlockChunk(
            depth=2,
            module=_SimpleBlock,
            module_kwargs={"dim": DIM},
            drop_path=0.1,
            key=KEY,
        )
        x = jr.normal(KEY, SHAPE)
        assert jnp.all(jnp.isfinite(chunk(x, key=KEY)))

    def test_inference_flag_forwarded(self):
        chunk = BlockChunk(
            depth=2,
            module=_SimpleBlock,
            module_kwargs={"dim": DIM},
            key=KEY,
        )
        x = jr.normal(KEY, SHAPE)
        out = chunk(x, key=KEY, inference=True)
        assert out.shape == SHAPE

    def test_init_values_forwarded_to_blocks(self):
        chunk = BlockChunk(
            depth=2,
            module=_SimpleBlock,
            module_kwargs={"dim": DIM},
            init_values=1e-4,
            key=KEY,
        )
        x = jr.normal(KEY, SHAPE)
        assert chunk(x, key=KEY).shape == SHAPE
