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
from equimo.models.vit import (
    dinov2_vits14_reg,
    dinov3_vits16_pretrain_lvd1689m,
    siglip2_vitb16_256,
    vit5_small,
)
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
        use_global_pos_embed=False,
        use_local_pos_embed=True,
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


# ConvNeXt


def test_convnext_forward():
    model = em.convnext_t(in_channels=3, num_classes=NUM_CLASSES, key=KEY)
    x = jr.normal(KEY, (3, 64, 64))
    y = model(x, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_convnext_features():
    model = em.convnext_t(in_channels=3, num_classes=0, key=KEY)
    x = jr.normal(KEY, (3, 64, 64))
    feats = model.features(x, key=KEY, inference=True)
    assert feats.ndim == 3  # (c, h, w)
    assert jnp.all(jnp.isfinite(feats))


def test_convnext_drop_path():
    model = em.ConvNeXt(
        in_channels=3,
        depths=[2, 2],
        dims=[32, 64],
        drop_path_rate=0.1,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    x = jr.normal(KEY, (3, 64, 64))
    y = model(x, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


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
    model = dinov2_vits14_reg(pretrained=True, dynamic_img_size=True)

    x = jr.normal(key, (3, 224, 224))
    features = model.features(x, key=key)

    assert features.shape[-1] == 384
    assert jnp.all(jnp.isfinite(features))


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
    model = dinov2_vits14_reg(pretrained=True)
    hash = _model_checksum(model)
    assert hash == "82bc53567e4565f6"

    fwd = model.forward_features(x, key=key, inference=True)
    eq_cls = np.array(fwd["x_norm_cls_token"])  # (384,)

    mae = float(np.mean(np.abs(eq_cls - ref["cls_token"])))
    assert mae < 1e-5, f"DINOv2 cls token MAE vs timm: {mae:.2e}"


def test_dinov3_vits16_matches_hf():
    """DINOv3 ViT-S/16 (LVD-1689M) cls token must match HuggingFace output.

    Reference features were extracted with:
        facebook/dinov3-vits16-pretrain-lvd1689m via transformers pipeline
    on a fixed random 256×256 image (see torch_models.py).

    Comparison:
    - HF  model(img).last_hidden_state[0, 0]       → post-norm cls token
    - equimo forward_features(x)["x_norm_cls_token"] → same quantity
    Tolerance: mean absolute error < 5e-4.
    """
    key = jr.PRNGKey(42)
    ref = np.load(Path(__file__).parent / "data" / "dinov3_vits16_reference.npz")

    x = jnp.array(ref["img"])  # (3, 256, 256)
    model = dinov3_vits16_pretrain_lvd1689m(pretrained=True)

    fwd = model.forward_features(x, key=key, inference=True)
    eq_cls = np.array(fwd["x_norm_cls_token"])  # (384,)

    mae = float(np.mean(np.abs(eq_cls - ref["cls_token"])))
    assert mae < 3e-4, f"DINOv3 cls token MAE vs HuggingFace: {mae:.2e}"


def test_siglip2_vitb16_256_matches_hf():
    """SigLIP2 ViT-B/16 at 256×256 patch tokens must match HuggingFace output.

    Reference features were extracted with:
        google/siglip2-base-patch16-256 vision_model via transformers pipeline
    on a fixed random 256×256 image (see torch_models.py).

    Comparison:
    - HF  vision_model(img).last_hidden_state[0]  → post-norm patch tokens (256, 768)
    - equimo jax.vmap(model.norm)(model.features(x)) → same quantity
    Tolerance: mean absolute error < 5e-4.
    """
    key = jr.PRNGKey(42)
    ref = np.load(Path(__file__).parent / "data" / "siglip2_vitb16_256_reference.npz")

    x = jnp.array(ref["img"])  # (3, 256, 256)
    model = siglip2_vitb16_256(pretrained=True)

    features = model.features(x, key=key, inference=True)
    eq_patches = np.array(jax.vmap(model.norm)(features))  # (256, 768)

    mae = float(np.mean(np.abs(eq_patches - ref["patch_tokens"])))
    assert mae < 1e-5, f"SigLIP2 patch tokens MAE vs HuggingFace: {mae:.2e}"


def test_eupe_vitt16_matches_torch():
    """EUPE ViT-T/16 prenorm features must match original PyTorch output.

    Reference features were extracted with PyTorch locally
    on a fixed random 224x224 image (see torch_models.py).

    Comparison:
    - PyTorch model.forward_features(x)["x_prenorm"]
    - equimo model.features(x)
    Tolerance: mean absolute error < 5e-4.
    """
    key = jr.PRNGKey(42)
    ref = np.load(Path(__file__).parent / "data" / "eupe_vitt16_reference.npz")

    x = jnp.array(ref["img"])  # (3, 224, 224)
    model = em.eupe_vitt16(pretrained=True)

    eq_features = np.array(model.features(x, key=key, inference=True))  # (201, 192)

    mae = float(np.mean(np.abs(eq_features - ref["features"][0])))
    assert mae < 5e-4, f"EUPE features MAE vs PyTorch: {mae:.2e}"


def test_vit5_small_forward():
    """ViT5-S/16 forward pass: correct output shape and finite values.

    Uses combined APE (patches only) + RoPE (patches + registers).
    """
    key = jr.PRNGKey(42)
    model = vit5_small(pretrained=False, key=key)
    x = jr.normal(key, (3, 224, 224))

    features = model.features(x, key=key, inference=True)

    # 5 prefix tokens (1 cls + 4 reg) + 196 patches
    assert features.shape == (201, 384)
    assert jnp.all(jnp.isfinite(features))


# DEQ
#
# DEQ-specific concerns that these tests exercise:
#   * The forward pass composes a fixed-point solve inside a regular forward,
#     so both inference and training modes must behave sanely.
#   * The solver must actually converge — ``aux["error"]`` must land below the
#     configured tolerance, and ``||f(z*, x) - z*||`` must be small.
#   * Gradients must flow through the NeumannPhantom adjoint (the main reason
#     to use a DEQ at all).
#   * Switching injector / stabilizer / strategy via kwargs must work.


def test_deq_forward():
    """Default preset (prenorm_add + projected + entry) — full forward pass."""
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    y, auxs = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))
    assert isinstance(auxs, list)
    # Default config has one FPI stage.
    assert len(auxs) == 1


def test_deq_forward_training_mode():
    """Training forward pass with DropPath enabled.

    The solver reuses a single RNG key across all Picard iterations to freeze
    stochastic ops; without this the fixed point does not exist. This test
    just verifies the training-mode forward runs and produces finite outputs.
    """
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        drop_path_rate=0.1,
        key=KEY,
    )
    y, auxs = model(IMG_64, key=KEY, inference=False)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))
    assert len(auxs) == 1


def test_deq_features():
    """features() returns a 3D feature map + the aux list."""
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=0,
        key=KEY,
    )
    feats, auxs = model.features(IMG_64, key=KEY, inference=True)
    assert feats.ndim == 3
    assert jnp.all(jnp.isfinite(feats))
    assert len(auxs) == 1


def test_deq_aux_structure():
    """Aux dict must expose everything downstream regularizers need."""
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        key=KEY,
    )
    _, auxs = model(IMG_64, key=KEY, inference=True)
    aux = auxs[0]
    for k in ("z_star", "trajectory", "depth", "error", "key", "x_context", "z0"):
        assert k in aux, f"aux dict missing required key: {k}"
    assert jnp.all(jnp.isfinite(aux["z_star"]))


def test_deq_solver_converges():
    """Solver must report convergence below tolerance and not burn max_steps."""
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        fpi_tol=1e-3,
        fpi_maxsteps=50,
        key=KEY,
    )
    _, auxs = model(IMG_64, key=KEY, inference=True)
    aux = auxs[0]
    # Solver tolerance is 1e-3; give an order of magnitude slack.
    assert float(aux["error"]) < 1e-2, f"solver error: {float(aux['error']):.2e}"
    # If depth ≈ max_steps, the solver ran out of budget without converging.
    assert int(aux["depth"]) < 50


def test_deq_fixed_point_consistency():
    """At convergence, ``f(z*, x) ≈ z*`` must hold within solver tolerance."""
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=0,
        key=KEY,
    )
    _, auxs = model.features(IMG_64, key=KEY, inference=True)
    aux = auxs[0]
    z_star = aux["z_star"]
    x_context = aux["x_context"]
    key_solve = aux["key"]

    # Locate the DEQ block (one FPI stage at index 2 in the default config).
    deq_stage_idx = next(
        i for i, blk in enumerate(model.blocks) if blk.deq_block is not None
    )
    cell = model.blocks[deq_stage_idx].deq_block.cell

    z_next = cell(z_star, x_context, inference=True, key=key_solve)
    rel = float(jnp.linalg.norm(z_next - z_star) / (jnp.linalg.norm(z_star) + 1e-8))
    assert rel < 1e-2, f"|f(z*) - z*| / |z*| = {rel:.2e}"


def test_deq_gradients_finite_and_nonzero():
    """Backward pass through DEQ must produce finite, non-zero gradients.

    This is the canonical DEQ smoke test: if the implicit-differentiation
    adjoint is broken, gradients will be NaN, inf, or uniformly zero.
    """
    import equinox as eqx

    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        key=KEY,
    )

    def loss_fn(m, x):
        y, _ = m(x, key=KEY, inference=True)
        return jnp.mean(y**2)

    grads = eqx.filter_grad(loss_fn)(model, IMG_64)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert len(leaves) > 0
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves), "NaN/Inf in gradients"
    assert any(jnp.any(g != 0) for g in leaves), "All-zero gradients"


def test_deq_determinism_same_key():
    """Same input and same key must produce the same output.

    Non-determinism here would mean the fixed point does not exist.
    """
    model = em.deq_convnext_t(in_channels=3, num_classes=NUM_CLASSES, key=KEY)
    y1, _ = model(IMG_64, key=KEY, inference=True)
    y2, _ = model(IMG_64, key=KEY, inference=True)
    assert jnp.allclose(y1, y2)


@pytest.mark.parametrize(
    "injector,stabilizer,strategy",
    [
        # Preset default: pre-norm injection + GroupNorm projection.
        ("prenorm_add", "projected", "entry"),
        # Pre-norm + damped projection.
        ("prenorm_add", "damped_projected", "entry"),
        # Projection stabilizer alone bounds the iterate even with trivial injection.
        ("add", "projected", "entry"),
        # Projection at every block.
        ("add", "projected", "per_block"),
        # Gated injector (init_gate=0.5) is a damped Picard at init;
        # projection on top gives a belt-and-suspenders setup.
        ("gated", "projected", "entry"),
        # Projection + damping with a projected forcing term.
        ("proj_add", "damped_projected", "entry"),
    ],
)
def test_deq_injector_stabilizer_strategy_combos(injector, stabilizer, strategy):
    """Each well-posed (injector, stabilizer, strategy) combo must converge.

    Combos that lack **any** bounding mechanism — e.g. ``add + identity`` or
    ``proj_add + damped`` — are deliberately absent: ConvNeXt blocks with
    ``LayerScale(1e-6)`` are near-identity, so without at least one of
    {pre-norm on z, projection on output, convex-mix gate with init_gate < 1}
    the Picard iteration ``z_{k+1} ≈ z_k + x`` diverges linearly and burns
    through ``max_steps``. That's expected behavior, not a bug — the minimal
    combo is the library's simple default for general use, and
    architecture-specific presets (like ``deq_convnext_t``) layer bounding on
    top. For a finiteness-only smoke matrix over all 32 combos, see
    ``test_implicit.py::test_deqblock_all_combinations_run``.
    """
    model = em.deq_convnext_t(
        in_channels=3,
        num_classes=NUM_CLASSES,
        fpi_injector=injector,
        fpi_stabilizer=stabilizer,
        fpi_strategy=strategy,
        key=KEY,
    )
    y, auxs = model(IMG_64, key=KEY, inference=True)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))
    # Every well-posed combo must converge within the solver's budget.
    assert int(auxs[0]["depth"]) < 50


def test_deq_blocks_is_tuple():
    """Structural invariant: BlockChunk.blocks must be a pytree-friendly tuple."""
    model = em.deq_convnext_t(in_channels=3, num_classes=NUM_CLASSES, key=KEY)
    assert isinstance(model.blocks, tuple)


def test_deq_jit_compatible():
    """The model must trace cleanly under filter_jit."""
    import equinox as eqx

    model = em.deq_convnext_t(in_channels=3, num_classes=NUM_CLASSES, key=KEY)

    @eqx.filter_jit
    def forward(m, x, k):
        y, _ = m(x, key=k, inference=True)
        return y

    y = forward(model, IMG_64, KEY)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_deq_get_fpi_cells():
    """get_fpi_cells() must return a tuple of DEQCell instances."""
    from equimo.implicit import DEQCell

    model = em.deq_convnext_t(in_channels=3, num_classes=NUM_CLASSES, key=KEY)
    cells = model.get_fpi_cells()
    
    assert isinstance(cells, tuple)
    assert len(cells) == 1
    assert isinstance(cells[0], DEQCell)
