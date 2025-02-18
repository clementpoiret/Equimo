from pathlib import Path

import jax

import equimo.models as em
from equimo.converters.utils import convert_torch_to_equinox
from equimo.io import load_model, save_model


def main():
    key = jax.random.PRNGKey(42)
    dinov2_vits14_reg_config = {
        "img_size": 518,
        "in_channels": 3,
        "dim": 384,
        "patch_size": 14,
        "num_heads": [6],
        "depths": [12],
        "num_classes": 0,
        "use_mask_token": True,
        "reg_tokens": 4,
        "init_values": 1e-5,
        "eps": 1e-6,
        "dynamic_img_size": True,
    }

    dinov2_vits14_reg = em.VisionTransformer(
        **dinov2_vits14_reg_config,
        key=key,
    )

    torch_hub_cfg = ["facebookresearch/dinov2", "dinov2_vits14_reg"]

    replace_cfg = {
        "reg_tokens": "register_tokens",
        "blocks.0.blocks": "blocks",
        ".prenorm.": ".norm1.",
        ".norm.": ".norm2.",
    }
    expand_cfg = {"patch_embed.proj.bias": ["after", 2]}
    squeeze_cfg = {
        "pos_embed": 0,
        "cls_token": 0,
        "register_tokens": 0,
    }
    whitelist = []

    dinov2_vits14_reg = convert_torch_to_equinox(
        dinov2_vits14_reg,
        torch_hub_cfg,
        replace_cfg,
        expand_cfg,
        squeeze_cfg,
        whitelist,
        strict=True,
    )

    save_model(
        Path("~/.cache/equimo/dinov2_vits14_reg").expanduser(),
        dinov2_vits14_reg,
        dinov2_vits14_reg_config,
        torch_hub_cfg,
        compression=True,
    )

    _ = load_model(
        cls="vit",
        path=Path("~/.cache/equimo/dinov2_vits14_reg.tar.lz4").expanduser(),
    )
