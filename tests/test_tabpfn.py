import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.conversion.utils import stringify_name
from equimo.tabular.loader import load_state_dict
from equimo.tabular.model import TabPFNV3, predict

KEY = jr.PRNGKey(0)


def _tiny(*, key=KEY, **overrides):
    cfg = dict(
        max_num_classes=4,
        embed_dim=16,
        dist_embed_num_blocks=1,
        dist_embed_num_heads=2,
        dist_embed_num_inducing_points=4,
        feature_group_size=3,
        feat_agg_num_blocks=2,
        feat_agg_num_heads=2,
        feat_agg_num_cls_tokens=2,
        nlayers=2,
        icl_num_heads=2,
        icl_num_kv_heads_test=1,
        decoder_head_dim=8,
        decoder_num_heads=2,
        ff_factor=2,
    )
    cfg.update(overrides)
    return TabPFNV3(**cfg, key=key)


def test_forward_shape_and_finite():
    m = _tiny()
    R, C, n_train = 12, 5, 8
    x = jr.normal(jr.PRNGKey(1), (R, C))
    y = jr.randint(jr.PRNGKey(2), (R,), 0, 4).astype(jnp.float32)
    out = predict(m, x, y, n_train)
    assert out.shape == (R - n_train, 4)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_forward_handles_nan_and_inf():
    m = _tiny()
    R, C, n_train = 12, 5, 8
    x = np.array(jr.normal(jr.PRNGKey(1), (R, C)))
    x[0, 0], x[3, 2], x[5, 1] = np.nan, np.inf, -np.inf
    y = jr.randint(jr.PRNGKey(2), (R,), 0, 4).astype(jnp.float32)
    out = predict(m, jnp.asarray(x), y, n_train)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_loader_roundtrip_maps_every_param():
    """A self-contained check that loader path<->key mapping is total: dump model
    A's params as a torch-style state_dict, load into a differently-initialised B,
    and require identical outputs."""
    a = _tiny(key=jr.PRNGKey(10))
    params, _ = jax.tree_util.tree_flatten_with_path(eqx.filter(a, eqx.is_array))
    state_dict = {stringify_name(p): np.asarray(leaf) for p, leaf in params}

    b = _tiny(key=jr.PRNGKey(999))
    b = load_state_dict(b, state_dict)  # strict=True: every leaf must be matched

    R, C, n_train = 12, 5, 8
    x = jr.normal(jr.PRNGKey(1), (R, C))
    y = jr.randint(jr.PRNGKey(2), (R,), 0, 4).astype(jnp.float32)
    assert jnp.allclose(predict(a, x, y, n_train), predict(b, x, y, n_train), atol=1e-6)


def test_loader_strict_rejects_extra_keys():
    a = _tiny(key=jr.PRNGKey(10))
    params, _ = jax.tree_util.tree_flatten_with_path(eqx.filter(a, eqx.is_array))
    sd = {stringify_name(p): np.asarray(leaf) for p, leaf in params}
    sd["bogus.extra.weight"] = np.zeros((2, 2), dtype="float32")
    with pytest.raises(KeyError):
        load_state_dict(_tiny(key=jr.PRNGKey(11)), sd, strict=True)
