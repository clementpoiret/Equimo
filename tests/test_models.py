import tempfile
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

import equimo.models as em
from equimo.io import save_model, load_model


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

    # Testing multiple img sizes, inference mode, and masking
    f1 = base_model.features(x1, mask=mask, inference=True, key=key)
    f2 = base_model.features(x2, inference=False, key=key)

    assert jnp.all(f1)
    assert jnp.all(f2)


def test_save_load_model_compressed():
    """Test saving and loading a model with compression."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a simple model
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

        # Create test input
        x = jr.normal(key, (3, 224, 224))
        original_output = model.features(x, key=key)

        # Save model
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

        # Load model
        loaded_model = load_model(
            cls="vit", path=save_path.with_suffix(".tar.lz4"), dynamic_img_size=True
        )

        # Test loaded model
        loaded_output = loaded_model.features(x, key=key)

        # Compare outputs
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

    # Test inference
    x = jr.normal(key, (3, 224, 224))
    features = model.features(x, key=key)

    assert features.shape[-1] == 384  # DINOv2-S has embedding dimension of 384
    assert jnp.all(jnp.isfinite(features))  # Check for NaN/Inf values


def test_reduceformer():
    """Test creation and inference of a ReduceFormer model."""
    key = jr.PRNGKey(42)

    x = jr.normal(key, (3, 64, 64))
    model = em.reduceformer_backbone_b1(in_channels=3, num_classes=10, key=key)
    y_hat = model(x, key=key)

    assert len(y_hat) == 10


def test_fused_reduceformer():
    """Test creation and inference of a ReduceFormer model with fused mbconv."""
    key = jr.PRNGKey(42)

    x = jr.normal(key, (3, 64, 64))
    model = em.reduceformer_backbone_b1(
        in_channels=3, num_classes=10, fuse_mbconv=True, key=key
    )
    y_hat = model(x, key=key)

    assert len(y_hat) == 10
