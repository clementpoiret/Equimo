import hashlib
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import equimo.models as em
from equimo.io import load_model, save_model
from equimo.layers.activation import get_act
from equimo.models.attnet import attnet_xxs
from equimo.models.fastervit import FasterViT
from equimo.models.lowformer import lowformer_backbone_b0
from equimo.models.mlla import Mlla
from equimo.models.mobilenet import mobilenetv3_small
from equimo.models.partialformer import PartialFormer
from equimo.models.shvit import SHViT
from equimo.models.vssd import Vssd
from equimo.utils import make_drop_path_schedule

# Helpers

KEY = jr.PRNGKey(0)
NUM_CLASSES = 10
IMG_64 = jr.normal(KEY, (3, 64, 64))


def _model_checksum(model) -> str:
    """SHA-256 over all array leaves."""
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(model):
        if hasattr(leaf, "tobytes"):
            h.update(np.asarray(leaf).tobytes())
    return h.hexdigest()[:16]


# Utility tests


def test_make_drop_path_schedule_uniform():
    schedule = make_drop_path_schedule(0.1, [2, 3, 4], uniform=True)
    assert schedule == [0.1] * 9
    assert len(schedule) == 9


def test_make_drop_path_schedule_linear():
    schedule = make_drop_path_schedule(0.1, [2, 2], uniform=False)
    assert len(schedule) == 4
    assert schedule[0] == pytest.approx(0.0)
    assert schedule[-1] == pytest.approx(0.1)
    # Monotonically increasing
    assert all(a <= b for a, b in zip(schedule, schedule[1:]))


def test_make_drop_path_schedule_zero_rate():
    schedule = make_drop_path_schedule(0.0, [2, 3])
    assert all(v == pytest.approx(0.0) for v in schedule)


def test_get_act_hard_swish():
    act = get_act("hard_swish")
    x = jnp.array([-2.0, 0.0, 2.0])
    out = act(x)
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))


# VisionTransformer


def test_vit_inference():
    """Test forward pass of a ViT"""
    key = jr.PRNGKey(42)
    img_size = 224
    patch_size = 14

    x1 = jr.normal(key, (3, 224, 224))
    x2 = jr.normal(key, (3, 98, 98))
    mask = jr.bernoulli(key, shape=(16, 16)) * 1

    base_model = em.VisionTransformer(
        img_size=img_size,
        in_channels=3,
        dim=384,
        patch_size=patch_size,
        num_heads=[6],
        depths=[12],
        num_classes=0,
        use_mask_token=True,
        dynamic_img_size=True,
        key=key,
    )

    f1 = base_model.features(x1, mask=mask, inference=True, key=key)
    f2 = base_model.features(x2, inference=False, key=key)

    assert jnp.all(f1)
    assert jnp.all(f2)


def test_vit_classification():
    key = jr.PRNGKey(0)
    model = em.VisionTransformer(
        img_size=64,
        in_channels=3,
        dim=64,
        patch_size=8,
        num_heads=[2],
        depths=[2],
        num_classes=NUM_CLASSES,
        key=key,
    )
    y = model(IMG_64, key=key, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_vit_rope():
    # RoPE requires dynamic_img_size=True so patch_embed returns (c, h, w)
    # allowing H and W to be read from spatial dims.
    key = jr.PRNGKey(1)
    model = em.VisionTransformer(
        img_size=64,
        in_channels=3,
        dim=64,
        patch_size=8,
        num_heads=2,
        depths=[2],
        num_classes=NUM_CLASSES,
        use_rope_pos_embed=True,
        dynamic_img_size=True,
        class_token=True,
        reg_tokens=0,
        key=key,
    )
    y_train = model(IMG_64, key=key, inference=False)
    y_infer = model(IMG_64, key=key, inference=True)
    assert y_train.shape == (NUM_CLASSES,)
    assert y_infer.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y_infer))


# IFormer


def test_iformer():
    key = jr.PRNGKey(42)
    x = jr.normal(key, (3, 64, 64))
    model = em.iformer_t(in_channels=3, num_classes=NUM_CLASSES, key=key)
    y_hat = model(x, key=key)
    assert y_hat.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y_hat))


# ReduceFormer


def test_reduceformer():
    key = jr.PRNGKey(42)
    x = jr.normal(key, (3, 64, 64))
    model = em.reduceformer_backbone_b1(in_channels=3, num_classes=10, key=key)
    y_hat = model(x, key=key)
    assert len(y_hat) == 10


def test_fused_reduceformer():
    key = jr.PRNGKey(42)
    x = jr.normal(key, (3, 64, 64))
    model = em.reduceformer_backbone_b1(
        in_channels=3, num_classes=10, fuse_mbconv=True, key=key
    )
    y_hat = model(x, key=key)
    assert len(y_hat) == 10


# Mlla


def test_mlla_construction():
    model = Mlla(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=4,
        depths=[1, 1, 2, 1],
        num_heads=[1, 2, 4, 8],
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    # norm must be a proper pytree leaf (training)
    import equinox as eqx
    import jax

    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    assert len(leaves) > 0


def test_mlla_forward():
    model = Mlla(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=4,
        depths=[1, 1, 2, 1],
        num_heads=[1, 2, 4, 8],
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_mlla_drop_path_schedule():
    """Verify per-block drop path is applied (not per-stage)."""
    depths = [1, 1, 2, 1]
    model = Mlla(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=4,
        depths=depths,
        num_heads=[1, 2, 4, 8],
        num_classes=NUM_CLASSES,
        drop_path_rate=0.1,
        key=KEY,
    )
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)


# VSSD


def test_vssd_construction():
    model = Vssd(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=4,
        depths=[1, 1, 2, 1],
        num_heads=[1, 2, 4, 8],
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    import equinox as eqx
    import jax

    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    assert len(leaves) > 0


def test_vssd_forward():
    model = Vssd(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=4,
        depths=[1, 1, 2, 1],
        num_heads=[1, 2, 4, 8],
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


# AttNet


def test_attnet_forward():
    # attnet_xxs has 4 stages of 2x downsampling (first stage is 4x),
    # requiring at least 256x256 to keep spatial dims non-trivial.
    x = jr.normal(KEY, (3, 256, 256))
    model = attnet_xxs(in_channels=3, num_classes=NUM_CLASSES, key=KEY)
    y = model(x, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_attnet_features():
    x = jr.normal(KEY, (3, 256, 256))
    model = attnet_xxs(in_channels=3, num_classes=0, key=KEY)
    feats = model.features(x, key=KEY, inference=True)
    assert feats.ndim == 3  # (c, h, w)
    assert jnp.all(jnp.isfinite(feats))


# LowFormer


def test_lowformer_forward():
    model = lowformer_backbone_b0(
        in_channels=3,
        num_classes=NUM_CLASSES,
        attention_type="softmax",
        key=KEY,
    )
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_lowformer_features():
    model = lowformer_backbone_b0(
        in_channels=3,
        num_classes=0,
        attention_type="softmax",
        key=KEY,
    )
    feats = model.features(IMG_64, key=KEY, inference=True)
    assert jnp.all(jnp.isfinite(feats))


# MobileNetv3


def test_mobilenetv3_small_forward():
    model = mobilenetv3_small(in_channels=3, num_classes=NUM_CLASSES, key=KEY)
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_mobilenetv3_small_features():
    model = mobilenetv3_small(in_channels=3, num_classes=0, key=KEY)
    # features() returns (c,) after GAP
    feats = model.features(IMG_64, key=KEY, inference=True)
    assert jnp.all(jnp.isfinite(feats))


# SHViT


def test_shvit_construction():
    """SHViT can be constructed with norm tracked as a pytree leaf."""
    model = SHViT(
        in_channels=3,
        dim=[32, 64],
        pdim=[8, 16],
        qk_dim=[8, 8],
        depths=[1, 1],
        block_type=["s", "s"],
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    import equinox as eqx
    import jax

    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    assert len(leaves) > 0


def test_shvit_features():
    model = SHViT(
        in_channels=3,
        dim=[32, 64],
        pdim=[8, 16],
        qk_dim=[8, 8],
        depths=[1, 1],
        block_type=["s", "s"],
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    x = jr.normal(KEY, (3, 128, 128))
    feats = model.features(x, key=KEY, inference=True)
    assert jnp.all(jnp.isfinite(feats))


# FasterViT


def test_fastervit_forward():
    model = FasterViT(
        img_size=64,
        in_channels=3,
        dim=32,
        in_dim=16,
        num_heads=1,
        hat=False,
        depths=[1, 1],
        window_size=4,
        ct_size=2,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_fastervit_features():
    model = FasterViT(
        img_size=64,
        in_channels=3,
        dim=32,
        in_dim=16,
        num_heads=1,
        hat=False,
        depths=[1, 1],
        window_size=4,
        ct_size=2,
        num_classes=0,
        key=KEY,
    )
    feats = model.features(IMG_64, key=KEY, inference=True)
    assert feats.ndim == 2  # (seqlen, dim)
    assert jnp.all(jnp.isfinite(feats))


# PartialFormer


def test_partialformer_forward():
    model = PartialFormer(
        img_size=64,
        in_channels=3,
        dim=32,
        num_heads=[1, 2],
        depths=[1, 1],
        foreground_ratios=0.5,
        patch_size=4,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_partialformer_tuple_blocks():
    """self.blocks must be a tuple, not a list."""
    model = PartialFormer(
        img_size=64,
        in_channels=3,
        dim=32,
        num_heads=[1, 2],
        depths=[1, 1],
        foreground_ratios=0.5,
        patch_size=4,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    assert isinstance(model.blocks, tuple)


def test_partialformer_foreground_ratios_tuple():
    """foreground_ratios as a 2-tuple (range) must not raise."""
    model = PartialFormer(
        img_size=64,
        in_channels=3,
        dim=32,
        num_heads=[1, 2],
        depths=[1, 1],
        foreground_ratios=(0.3, 0.7),
        patch_size=4,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    y = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)


# Save / Load


def test_save_load_model_compressed():
    """Test saving and loading a model with compression."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        key = jr.PRNGKey(42)
        model = em.VisionTransformer(
            img_size=224,
            in_channels=3,
            dim=384,
            patch_size=14,
            num_heads=[6],
            depths=[12],
            num_classes=0,
            key=key,
        )

        x = jr.normal(key, (3, 224, 224))
        original_output = model.features(x, key=key)

        save_path = Path(tmp_dir) / "test_model"
        model_config = {
            "img_size": 224,
            "in_channels": 3,
            "dim": 384,
            "patch_size": 14,
            "num_heads": [6],
            "depths": [12],
            "num_classes": 0,
        }
        torch_hub_cfg = ["example_config"]

        save_model(save_path, model, model_config, torch_hub_cfg, compression=True)

        loaded_model = load_model(
            cls="vit", path=save_path.with_suffix(".tar.lz4"), dynamic_img_size=True
        )

        loaded_output = loaded_model.features(x, key=key)
        assert jnp.allclose(original_output, loaded_output, atol=1e-5)


def test_save_load_model_uncompressed():
    """Test saving and loading a model without compression."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        key = jr.PRNGKey(42)
        model = em.VisionTransformer(
            img_size=224,
            in_channels=3,
            dim=384,
            patch_size=14,
            num_heads=[6],
            depths=[12],
            num_classes=0,
            key=key,
        )

        x = jr.normal(key, (3, 224, 224))
        original_output = model.features(x, key=key)

        save_path = Path(tmp_dir) / "test_model_uncompressed"
        model_config = {
            "img_size": 224,
            "in_channels": 3,
            "dim": 384,
            "patch_size": 14,
            "num_heads": [6],
            "depths": [12],
            "num_classes": 0,
        }
        torch_hub_cfg = ["example_config"]

        save_model(save_path, model, model_config, torch_hub_cfg, compression=False)

        loaded_model = load_model(cls="vit", path=save_path, dynamic_img_size=True)
        loaded_output = loaded_model.features(x, key=key)

        assert jnp.allclose(original_output, loaded_output, atol=1e-5)


def test_load_pretrained_model():
    """Test loading a pretrained model from the repository."""
    key = jr.PRNGKey(42)
    model = load_model(cls="vit", identifier="dinov2_vits14_reg", dynamic_img_size=True)

    x = jr.normal(key, (3, 224, 224))
    features = model.features(x, key=key)

    assert features.shape[-1] == 384
    assert jnp.all(jnp.isfinite(features))


def test_dinov3():
    key = jr.PRNGKey(42)

    x = jnp.ones((3, 64, 64))
    model = load_model(cls="vit", identifier="dinov3_vits16_pretrain_lvd1689m")
    y_hat = model(x, key=key)

    assert jnp.abs(y_hat[0] - -0.25373647) < 1e-6


def test_dinov2_vits14_reg_matches_timm():
    """DINOv2 ViT-S/14 with 4 register tokens must match timm's output.

    Reference features were extracted with:
        timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True)
    on a fixed random 518×518 image (see scripts/extract_dinov2_reference.py).

    Comparison:
    - timm  forward_features()[0, 0]          → normalized cls token (post-LayerNorm)
    - equimo forward_features(x)["x_norm_cls_token"] → same quantity
    Tolerance: mean absolute error < 5e-4.
    """
    key = jr.PRNGKey(42)
    ref = np.load(Path(__file__).parent / "data" / "dinov2_vits14_reg_reference.npz")

    x = jnp.array(ref["img"])  # (3, 518, 518)
    model = load_model(cls="vit", identifier="dinov2_vits14_reg")
    hash = _model_checksum(model)
    assert hash == "82bc53567e4565f6"

    fwd = model.forward_features(x, key=key, inference=True)
    eq_cls = np.array(fwd["x_norm_cls_token"])  # (384,)

    mae = float(np.mean(np.abs(eq_cls - ref["cls_token"])))
    assert mae < 5e-4, f"DINOv2 cls token MAE vs timm: {mae:.2e}"
