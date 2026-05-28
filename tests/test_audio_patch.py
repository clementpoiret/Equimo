import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.audio.layers import SpectrogramPatchEmbedding
from equimo.audio.layers.patch import (
    _PATCH_REGISTRY,
    get_patch,
    register_patch,
)


KEY = jr.PRNGKey(0)


class TestSpectrogramPatchEmbedding:
    DIM = 16
    INPUT_FDIM = 32
    INPUT_TDIM = 64
    PATCH_SIZE = 16

    def test_output_shape(self):
        layer = SpectrogramPatchEmbedding(
            dim=self.DIM,
            patch_size=self.PATCH_SIZE,
            input_fdim=self.INPUT_FDIM,
            input_tdim=self.INPUT_TDIM,
            fstride=16,
            tstride=16,
            key=KEY,
        )
        x = jr.normal(KEY, (self.INPUT_TDIM, self.INPUT_FDIM))

        assert layer(x).shape == (8, self.DIM)

    def test_metadata(self):
        layer = SpectrogramPatchEmbedding(
            dim=self.DIM,
            patch_size=(8, 16),
            input_fdim=self.INPUT_FDIM,
            input_tdim=self.INPUT_TDIM,
            fstride=8,
            tstride=16,
            key=KEY,
        )

        assert layer.patch_size == (8, 16)
        assert layer.stride == (8, 16)
        assert layer.img_size == (self.INPUT_FDIM, self.INPUT_TDIM)
        assert layer.grid_size == (4, 4)
        assert layer.num_patches == 16

    def test_output_finite(self):
        layer = SpectrogramPatchEmbedding(
            dim=self.DIM,
            patch_size=self.PATCH_SIZE,
            input_fdim=self.INPUT_FDIM,
            input_tdim=self.INPUT_TDIM,
            fstride=16,
            tstride=16,
            key=KEY,
        )
        x = jr.normal(KEY, (self.INPUT_TDIM, self.INPUT_FDIM))

        assert jnp.all(jnp.isfinite(layer(x)))

    def test_dtype_preserved_bfloat16(self):
        layer = SpectrogramPatchEmbedding(
            dim=self.DIM,
            patch_size=self.PATCH_SIZE,
            input_fdim=self.INPUT_FDIM,
            input_tdim=self.INPUT_TDIM,
            fstride=16,
            tstride=16,
            key=KEY,
        )
        layer = jax.tree_util.tree_map(
            lambda leaf: (
                leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf
            ),
            layer,
        )
        x = jr.normal(KEY, (self.INPUT_TDIM, self.INPUT_FDIM)).astype(jnp.bfloat16)

        assert layer(x).dtype == jnp.bfloat16

    def test_wrong_time_raises(self):
        layer = SpectrogramPatchEmbedding(
            dim=self.DIM,
            patch_size=self.PATCH_SIZE,
            input_fdim=self.INPUT_FDIM,
            input_tdim=self.INPUT_TDIM,
            fstride=16,
            tstride=16,
            key=KEY,
        )

        with pytest.raises(AssertionError, match="time dimension"):
            layer(jnp.ones((self.INPUT_TDIM + 1, self.INPUT_FDIM)))

    def test_wrong_frequency_raises(self):
        layer = SpectrogramPatchEmbedding(
            dim=self.DIM,
            patch_size=self.PATCH_SIZE,
            input_fdim=self.INPUT_FDIM,
            input_tdim=self.INPUT_TDIM,
            fstride=16,
            tstride=16,
            key=KEY,
        )

        with pytest.raises(AssertionError, match="frequency dimension"):
            layer(jnp.ones((self.INPUT_TDIM, self.INPUT_FDIM + 1)))

    def test_patch_too_large_raises(self):
        with pytest.raises(ValueError, match="Patch size"):
            SpectrogramPatchEmbedding(
                dim=self.DIM,
                patch_size=64,
                input_fdim=self.INPUT_FDIM,
                input_tdim=self.INPUT_TDIM,
                fstride=16,
                tstride=16,
                key=KEY,
            )


def test_audio_patch_registry_get_patch():
    assert get_patch("spectrogrampatchembedding") is SpectrogramPatchEmbedding
    assert get_patch(SpectrogramPatchEmbedding) is SpectrogramPatchEmbedding


def test_audio_patch_registry_duplicate_raises():
    with pytest.raises(ValueError, match="already registered"):

        @register_patch("spectrogrampatchembedding")
        class DuplicateSpectrogramPatchEmbedding(eqx.Module):
            pass


def test_audio_patch_registry_contains_builtin():
    assert _PATCH_REGISTRY["spectrogrampatchembedding"] is SpectrogramPatchEmbedding
