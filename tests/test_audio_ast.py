import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.audio.models as am
from equimo.registry import get_model_cls


KEY = jr.PRNGKey(0)
NUM_CLASSES = 10
SPEC = jr.normal(KEY, (64, 32))


def _small_ast(**kwargs):
    cfg = {
        "input_fdim": 32,
        "input_tdim": 64,
        "dim": 32,
        "patch_size": 16,
        "fstride": 16,
        "tstride": 16,
        "num_heads": 4,
        "depths": [1],
        "num_classes": NUM_CLASSES,
        "key": KEY,
    }
    cfg.update(kwargs)
    return am.AudioSpectrogramTransformer(**cfg)


def test_ast_forward():
    model = _small_ast()
    y = model(SPEC, key=KEY, inference=True)

    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_ast_features_and_aux_shapes():
    model = _small_ast()
    features = model.features(SPEC, key=KEY, inference=True)
    aux = model.forward_features(SPEC, key=KEY, inference=True)

    assert model.grid_size == (2, 4)
    assert model.num_patches == 8
    assert features.shape == (10, 32)
    assert aux["x_norm_cls_token"].shape == (32,)
    assert aux["x_norm_dist_token"].shape == (32,)
    assert aux["x_norm_patchtokens"].shape == (8, 32)
    assert aux["x_prenorm"].shape == (10, 32)
    assert jnp.all(jnp.isfinite(features))


def test_ast_batching_with_vmap():
    model = _small_ast()
    batch = jr.normal(KEY, (3, 64, 32))

    y = jax.vmap(lambda x: model(x, key=KEY, inference=True))(batch)

    assert y.shape == (3, NUM_CLASSES)
    assert jnp.all(jnp.isfinite(y))


def test_ast_factory_forward_with_overrides():
    model = am.ast_tiny_patch16_224(
        input_fdim=32,
        input_tdim=64,
        dim=32,
        fstride=16,
        tstride=16,
        num_heads=4,
        depths=[1],
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    y = model(SPEC, key=KEY, inference=True)

    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_audio_ast_registered_by_modality():
    assert get_model_cls("ast", modality="audio") is am.AudioSpectrogramTransformer


def test_ast_wrong_input_size_raises():
    model = _small_ast()

    with pytest.raises(AssertionError, match="time dimension"):
        model(jnp.ones((65, 32)), key=KEY, inference=True)

    with pytest.raises(AssertionError, match="frequency dimension"):
        model(jnp.ones((64, 33)), key=KEY, inference=True)
