import importlib

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.registry import get_model_cls
from equimo.tabular import layers
from equimo.tabular.loader import build_model
from equimo.tabular.models import TabPFN, tabpfn

KEY = jr.PRNGKey(0)


def _tiny(*, key=KEY, **overrides):
    cfg = dict(
        num_classes=4,
        dim=16,
        depths=(1, 2, 2),
        num_heads=(2, 2, 2),
        num_inducing_points=4,
        feature_group_size=3,
        num_cls_tokens=2,
        num_kv_heads_test=1,
        decoder_head_dim=8,
        decoder_num_heads=2,
        mlp_ratio=2.0,
    )
    cfg.update(overrides)
    return TabPFN(**cfg, key=key)


def _inputs():
    rows, columns, n_train = 12, 5, 8
    x = jr.normal(jr.PRNGKey(1), (rows, columns))
    y = jr.randint(jr.PRNGKey(2), (rows,), 0, 4)
    return x, y, n_train


def test_forward_shape_and_finite():
    model = _tiny()
    x, y, n_train = _inputs()
    out = model(x, y, n_train, key=KEY, inference=True)
    assert out.shape == (x.shape[0] - n_train, 4)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_features_and_forward_features_shapes():
    model = _tiny()
    x, y, n_train = _inputs()
    features = model.features(x, y, n_train, key=KEY, inference=True)
    forward = model.forward_features(x, y, n_train, key=KEY, inference=True)

    assert features.shape == (x.shape[0], model.context_dim)
    assert forward["x_prenorm"].shape == features.shape
    assert forward["x_norm_train"].shape == (n_train, model.context_dim)
    assert forward["x_norm_test"].shape == (x.shape[0] - n_train, model.context_dim)
    assert bool(jnp.all(jnp.isfinite(features)))
    assert bool(jnp.all(jnp.isfinite(forward["x_norm_test"])))


def test_forward_handles_nan_and_inf():
    model = _tiny()
    x, y, n_train = _inputs()
    x = np.array(x)
    x[0, 0], x[3, 2], x[5, 1] = np.nan, np.inf, -np.inf
    out = model(jnp.asarray(x), y, n_train, key=KEY, inference=True)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_factory_and_registry():
    model = tabpfn(
        num_classes=4,
        dim=16,
        depths=(1, 1, 1),
        num_heads=(2, 2, 2),
        num_inducing_points=4,
        num_cls_tokens=2,
        decoder_head_dim=8,
        decoder_num_heads=2,
        key=KEY,
    )
    assert isinstance(model, TabPFN)
    assert get_model_cls("tabpfn", modality="tabular") is TabPFN


def test_constructor_resolves_tabular_layer_names():
    model = _tiny(
        context_block="incontextattentionblock",
        preprocessor_layer="preprocessor",
        label_embedding_layer="labelembedding",
        decoder_layer="attentiondecoder",
    )

    assert isinstance(model.preprocessor, layers.Preprocessor)
    assert isinstance(model.column_label_embedding, layers.LabelEmbedding)
    assert isinstance(model.context_label_embedding, layers.LabelEmbedding)
    assert isinstance(model.head, layers.AttentionDecoder)
    assert isinstance(model.blocks[0].blocks[0], layers.InContextAttentionBlock)
    assert not hasattr(model, "column_target_embedding")
    assert not hasattr(model, "context_target_embedding")


def test_context_drop_path_schedule():
    model = _tiny(depths=(1, 1, 3), drop_path_rate=0.3)
    blocks = model.blocks[0].blocks

    assert [block.drop_path1.p for block in blocks] == pytest.approx(
        [0.0, 0.15, 0.3]
    )
    assert [block.drop_path2.p for block in blocks] == pytest.approx(
        [0.0, 0.15, 0.3]
    )

    uniform = _tiny(depths=(1, 1, 3), drop_path_rate=0.3, drop_path_uniform=True)
    assert [block.drop_path1.p for block in uniform.blocks[0].blocks] == pytest.approx(
        [0.3, 0.3, 0.3]
    )


def test_loader_build_model_accepts_new_constructor_keys():
    model = build_model(
        {
            "num_classes": 4,
            "dim": 16,
            "depths": (1, 1, 2),
            "num_heads": (2, 2, 2),
            "num_inducing_points": 4,
            "num_cls_tokens": 2,
            "decoder_head_dim": 8,
            "decoder_num_heads": 2,
            "drop_path_rate": 0.2,
            "drop_path_uniform": True,
            "context_block": "incontextattentionblock",
            "preprocessor_layer": "preprocessor",
            "label_embedding_layer": "labelembedding",
            "decoder_layer": "attentiondecoder",
        },
        key=KEY,
    )

    assert isinstance(model, TabPFN)
    assert [block.drop_path1.p for block in model.blocks[0].blocks] == pytest.approx(
        [0.2, 0.2]
    )


def test_tabular_packages_import():
    importlib.import_module("equimo.tabular")
    importlib.import_module("equimo.tabular.models")
    importlib.import_module("equimo.tabular.layers")
