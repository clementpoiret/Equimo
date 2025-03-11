from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import equimo.models as em
from equimo.conversion.utils import convert_torch_to_equinox
from equimo.io import save_model


def compare(j, t) -> float:
    j = np.array(j)
    t = t.squeeze().detach().numpy()
    return float(np.mean(np.abs(j - t)))


configs = {
    "dinov2_vits14_reg": {
        "dim": 384,
        "num_heads": [6],
        "depths": [12],
        "reg_tokens": 4,
    },
    "dinov2_vits14": {
        "dim": 384,
        "num_heads": [6],
        "depths": [12],
        "reg_tokens": 0,
    },
    "dinov2_vitb14_reg": {
        "dim": 768,
        "num_heads": [12],
        "depths": [12],
        "reg_tokens": 4,
    },
    "dinov2_vitb14": {
        "dim": 768,
        "num_heads": [12],
        "depths": [12],
        "reg_tokens": 0,
    },
    "dinov2_vitl14_reg": {
        "dim": 1024,
        "num_heads": [16],
        "depths": [24],
        "reg_tokens": 4,
    },
    "dinov2_vitl14": {
        "dim": 1024,
        "num_heads": [16],
        "depths": [24],
        "reg_tokens": 0,
    },
    "dinov2_vitg14_reg": {
        "dim": 1536,
        "num_heads": [24],
        "depths": [40],
        "reg_tokens": 4,
        "ffn_layer": "swiglufused",
    },
    "dinov2_vitg14": {
        "dim": 1536,
        "num_heads": [24],
        "depths": [40],
        "reg_tokens": 0,
        "ffn_layer": "swiglufused",
    },
}

citr = iter(configs.items())
name, config = next(citr)


def main():
    try:
        import torch
    except:
        raise ImportError("`torch` not available")

    key = jax.random.PRNGKey(42)
    dinov2_config = {
        "img_size": 518,
        "in_channels": 3,
        # "dim": 384,
        "patch_size": 14,
        # "num_heads": [6],
        # "depths": [12],
        "num_classes": 0,
        "use_mask_token": True,
        # "reg_tokens": 4,
        "init_values": 1e-5,
        "eps": 1e-6,
        "dynamic_img_size": False,
        "act_layer": "exactgelu",
    }

    for name, config in configs.items():
        print(f"Converting {name}...")

        cfg = dinov2_config | config

        dinov2 = em.VisionTransformer(
            **cfg,
            key=key,
        )

        torch_hub_cfg = ["facebookresearch/dinov2", name]

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

        dinov2, torch_model = convert_torch_to_equinox(
            dinov2,
            replace_cfg,
            expand_cfg,
            squeeze_cfg,
            whitelist,
            strict=True,
            torch_hub_cfg=torch_hub_cfg,
            return_torch=True,
        )

        arr = np.random.randn(3, cfg["img_size"], cfg["img_size"])
        jax_arr = jnp.array(arr)
        torch_arr = torch.tensor(arr).unsqueeze(0).float()

        assert (
            compare(
                dinov2.features(jax_arr, key),
                torch_model.forward_features(torch_arr)["x_prenorm"],
            )
            < 1e-5
        )

        save_model(
            Path(f"~/.cache/equimo/dinov2/{name}").expanduser(),
            dinov2,
            cfg,
            torch_hub_cfg,
            compression=True,
        )
