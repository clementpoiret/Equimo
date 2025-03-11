from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import equimo.models as em
from equimo.conversion.utils import convert_torch_to_equinox
from equimo.io import save_model
from tips.pytorch import image_encoder

CKPT_PATHS = {
    "tips_vits14_hr": "/mnt/hdd/torch/tips/tips_oss_s14_highres_distilled_vision.npz",
    "tips_vitb14_hr": "/mnt/hdd/torch/tips/tips_oss_b14_highres_distilled_vision.npz",
    "tips_vitl14_hr": "/mnt/hdd/torch/tips/tips_oss_l14_highres_distilled_vision.npz",
    "tips_vitso400m14_hr": "/mnt/hdd/torch/tips/tips_oss_so400m14_highres_largetext_distilled_vision.npz",
    "tips_vitg14_lr": "/mnt/hdd/torch/tips/tips_oss_g14_lowres_vision.npz",
    "tips_vitg14_hr": "/mnt/hdd/torch/tips/tips_oss_g14_highres_vision.npz",
}

CLS = {
    "tips_vits14_hr": image_encoder.vit_small,
    "tips_vitb14_hr": image_encoder.vit_base,
    "tips_vitl14_hr": image_encoder.vit_large,
    "tips_vitso400m14_hr": image_encoder.vit_so400m,
    "tips_vitg14_lr": image_encoder.vit_giant2,
    "tips_vitg14_hr": image_encoder.vit_giant2,
}


def compare(j, t) -> float:
    j = np.array(j)
    t = t.squeeze().detach().numpy()
    return float(np.mean(np.abs(j - t)))


configs = {
    "tips_vits14_hr": {
        "img_size": 448,
        "dim": 384,
        "num_heads": [6],
        "depths": [12],
    },
    "tips_vitb14_hr": {
        "img_size": 448,
        "dim": 768,
        "num_heads": [12],
        "depths": [12],
    },
    "tips_vitl14_hr": {
        "img_size": 448,
        "dim": 1024,
        "num_heads": [16],
        "depths": [24],
    },
    "tips_vitso400m14_hr": {
        "img_size": 448,
        "dim": 1152,
        "num_heads": [16],
        "depths": [27],
        "mlp_ratio": 4304 / 1152,
    },
    "tips_vitg14_lr": {
        "img_size": 224,
        "dim": 1536,
        "num_heads": [24],
        "depths": [40],
        "ffn_layer": "swiglufused",
    },
    "tips_vitg14_hr": {
        "img_size": 448,
        "dim": 1536,
        "num_heads": [24],
        "depths": [40],
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
    base_config = {
        # "img_size": 448,
        "in_channels": 3,
        # "dim": 384,
        "patch_size": 14,
        # "num_heads": [6],
        # "depths": [12],
        "num_classes": 0,
        "use_mask_token": True,
        "reg_tokens": 1,
        "init_values": 1e-5,
        "eps": 1e-6,
        "dynamic_img_size": False,
        "act_layer": "exactgelu",
    }

    for name, config in configs.items():
        print(f"Converting {name}...")

        cfg = base_config | config

        tips = em.VisionTransformer(
            **cfg,
            key=key,
        )

        weights_image = dict(np.load(CKPT_PATHS[name], allow_pickle=False))
        for k in weights_image:
            weights_image[k] = torch.tensor(weights_image[k])

        with torch.no_grad():
            # Load the vision encoder.
            model_image = CLS[name](
                img_size=224 if "lr" in name else 448,
                patch_size=14,
                ffn_layer="swiglu" if "vitg" in name else "mlp",
                block_chunks=0,
                init_values=1.0,
                interpolate_antialias=True,
                interpolate_offset=0.0,
            )
            model_image.load_state_dict(weights_image)

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

        tips, torch_model = convert_torch_to_equinox(
            tips,
            replace_cfg,
            expand_cfg,
            squeeze_cfg,
            whitelist,
            strict=True,
            source="custom",
            torch_model=model_image,
            return_torch=True,
        )

        arr = np.random.randn(3, cfg["img_size"], cfg["img_size"])
        jax_arr = jnp.array(arr)
        torch_arr = torch.tensor(arr).unsqueeze(0).float()

        assert (
            compare(
                tips.features(jax_arr, key),
                torch_model.forward_features(torch_arr)["x_prenorm"],
            )
            < 1e-5
        )

        save_model(
            Path(f"~/.cache/equimo/tips/{name}").expanduser(),
            tips,
            cfg,
            compression=True,
        )
